import argparse
import itertools
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from nits.model import *
from nits.layer import *
from nits.fc_model import *
from nits.cnn_model import *
from maf.datasets import *
from nits.resmade import ResidualMADE
from nits.fc_model import ResMADEModel
from scipy.stats import gaussian_kde
from scipy.special import kl_div
import os
from pathlib import Path
import glob
from antenna_utils import standard_scaler, create_dataloader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def correlation_dist(corr_mat1, corr_mat2):
    d = 1 - np.trace(np.dot(corr_mat1, corr_mat2)) / (np.linalg.norm(corr_mat1) * np.linalg.norm(corr_mat2))
    return d


def kl_eval(model, data, n):
    paths = sorted(glob.glob('models/ANTmodel_*.pth'))
    vv = np.linspace(-4, 4, num=7000)
    kl_divs = []
    for path in paths:
        kl_divs_path = []
        print(path)
        model.load_state_dict(torch.load(path, map_location=device))
        smp = model.model.sample(n, device)
        for feature in range(smp.shape[1]):
            kde_smp = gaussian_kde(smp[:, feature]).pdf(vv)
            kde_real = gaussian_kde(data.trn.x[:, feature]).pdf(vv)
            kl_div_value = np.sum(kl_div(kde_real, kde_smp))
            kl_divs_path.append(kl_div_value)
        kl_divs.append(kl_divs_path)
    kl_divs = np.array(kl_divs)
    kl_divs = np.sum(kl_divs, axis=1)
    best_path = paths[np.argmin(kl_divs)]
    print(f'best model for KL Divergence evaluation is {best_path}')
    return


def correlation_eval(model, data, n):
    paths = sorted(glob.glob('models/ANTmodel_*.pth'))
    distances = []
    for path in paths:
        print(path)
        model.load_state_dict(torch.load(path, map_location=device))
        smp = model.model.sample(n, device)
        corr_real, corr_smp = calc_corr_cov(smp, data)
        dist = correlation_dist(corr_real, corr_smp)
        distances.append(dist)
    best_path = paths[np.argmin(distances)]
    print(f'best model for correlation matrices evaluation is {best_path}')
    return


def calc_corr_cov(smp, data, path_to_save=None):
    n = len(smp)
    real = torch.Tensor(data.trn.x)
    corr_smp = np.corrcoef(smp.cpu().detach().numpy().T)
    plt.figure()
    plt.imshow(corr_smp, interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation matrix of generated samples')
    if path_to_save is not None:
        plt.savefig(f'{path_to_save}/correlation_matrices_generated.png')
    plt.figure()
    corr_real = np.corrcoef(real[:n].cpu().detach().numpy().T)
    plt.imshow(corr_real, interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation matrix of real samples')
    if path_to_save is not None:
        plt.savefig(f'{path_to_save}/correlation_matrices_real.png')
    plt.figure()
    diff = np.abs(corr_smp - corr_real)
    plt.imshow(diff, interpolation='nearest')
    plt.colorbar()
    plt.title('Difference between Correlation matrices')
    if path_to_save is not None:
        plt.savefig(f'{path_to_save}/correlation_matrices_difference.png')
    return corr_real, corr_smp


def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s


def create_batcher(x, batch_size=1):
    idx = 0
    p = torch.randperm(len(x))
    x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx + batch_size], device=device)
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device)


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='antenna')
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('-hi', '--hidden_dim', type=int, default=128)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=8)
parser.add_argument('-n', '--patience', type=int, default=10)
parser.add_argument('-ga', '--gamma', type=float, default=0.9)
parser.add_argument('-pd', '--polyak_decay', type=float, default=0.995)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-r', '--rotate', action='store_true')
parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
parser.add_argument('-l', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-p', '--dropout', type=float, default=-1.0)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=True)
parser.add_argument('-bm', '--bound_multiplier', type=int, default=1)
parser.add_argument('-dq', '--dequantize', type=bool, default=False,
                    help='do we dequantize the pixels? performs uniform dequantization')
parser.add_argument('-ds', '--discretized', type=bool, default=False,
                    help='Discretized model')
parser.add_argument('-w', '--step_weights', type=list_str_to_list, default='[1]',
                    help='Weights for each step of multistep NITS')
parser.add_argument('--scarf', action="store_true")
parser.add_argument('--bounds', type=list_str_to_list, default='[-10,10]')
parser.add_argument('--conditional_dim', type=int, default=12)
args = parser.parse_args()
conditional = args.conditional_dim is not None
lr_grid = [5e-3]
hidden_dim_grid = [256]
nr_blocks_grid = [8]
polyak_decay_grid = [0.9]
batch_size_grid = [20]
max_vals_ll = []
lasts_train_ll = []
model_names = []
for lr, hidden_dim, nr_blocks, polyak_decay, bs in itertools.product(lr_grid, hidden_dim_grid, nr_blocks_grid,
                                                                     polyak_decay_grid, batch_size_grid):
    model_extra_string = f'lr_{lr}_hd_{hidden_dim}_nr_{nr_blocks}_pd_{polyak_decay}_bs_{bs}'
    model_names.append(model_extra_string)
    print(model_extra_string)
    args.learning_rate = lr
    args.hidden_dim = hidden_dim
    args.n_residual_blocks = nr_blocks
    args.polyak_decay = polyak_decay
    args.batch_size = bs
    step_weights = np.array(args.step_weights)
    step_weights = step_weights / (np.sum(step_weights) + 1e-7)

    device = 'cuda:' + args.gpu if args.gpu else 'cpu'

    use_batch_norm = True
    zero_initialization = False
    weight_norm = False
    default_patience = 10
    assert args.dataset == 'antenna'
    data_path = r'../etof_folder_git/AntennaDesign_data/newdata_dB.npz'
    assert os.path.exists(data_path)
    data_tmp = np.load(data_path)
    data = miniboone.MINIBOONE()
    data.Data = data_tmp
    data.n_dims = data_tmp['parameters_train'].shape[1]
    scaler = standard_scaler()
    scaler.fit(data_tmp['parameters_train'])
    train_params_scaled = scaler.forward(data_tmp['parameters_train'])
    val_params_scaled = scaler.forward(data_tmp['parameters_val'])
    test_params_scaled = scaler.forward(data_tmp['parameters_test'])
    data.trn.x, data.trn.gamma,data.trn.radiation = train_params_scaled.astype(np.float32), data_tmp['gamma_train'], data_tmp['radiation_train']
    data.val.x, data.val.gamma,data.val.radiation = val_params_scaled.astype(np.float32), data_tmp['gamma_val'], data_tmp['radiation_val']
    data.tst.x, data.tst.gamma,data.tst.radiation = test_params_scaled.astype(np.float32), data_tmp['gamma_test'], data_tmp['radiation_test']
    default_dropout = 0
    args.patience = args.patience if args.patience >= 0 else default_patience
    args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
    print(args)

    d = data.trn.x.shape[1]

    max_val = args.bounds[1]  # max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
    min_val = args.bounds[0]  # min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
    max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

    max_val *= args.bound_multiplier
    min_val *= args.bound_multiplier
    nits_input_dim = [1]
    nits_model = NITS(d=d, arch=nits_input_dim + args.nits_arch, start=min_val, end=max_val, monotonic_const=1e-5,
                      A_constraint='neg_exp',
                      final_layer_constraint='softmax',
                      add_residual_connections=args.add_residual_connections,
                      normalize_inverse=(not args.dont_normalize_inverse),
                      softmax_temperature=False).to(device)

    model = Model(
        d=d,
        rotate=args.rotate,
        nits_model=nits_model,
        n_residual_blocks=args.n_residual_blocks,
        hidden_dim=args.hidden_dim,
        dropout_probability=args.dropout,
        use_batch_norm=use_batch_norm,
        zero_initialization=zero_initialization,
        weight_norm=weight_norm,
        nits_input_dim=nits_input_dim,
        conditional=conditional,
        conditional_dim=args.conditional_dim).to(device)

    shadow = Model(
        d=d,
        rotate=args.rotate,
        nits_model=nits_model,
        n_residual_blocks=args.n_residual_blocks,
        hidden_dim=args.hidden_dim,
        dropout_probability=args.dropout,
        use_batch_norm=use_batch_norm,
        zero_initialization=zero_initialization,
        weight_norm=weight_norm,
        nits_input_dim=nits_input_dim,
        conditional=conditional,
        conditional_dim=args.conditional_dim).to(device)

    # initialize weight norm
    if weight_norm:
        with torch.no_grad():
            for i, (gamma,rad,x) in enumerate(create_dataloader(data.trn.gamma,data.trn.radiation,data.trn.x,
                                                                batch_size=args.batch_size, device=device)):
                params = model(x,(gamma,rad))
                break

    model = EMA(model, shadow, decay=args.polyak_decay).to(device)
    # n = 5000
    # vv = np.linspace(-4, 4, num=7000)
    # path = 'models\\ANT_model_lr_0.0005_hd_256_nr_8_pd_0.9_bs_20.pth'
    # distances = []
    # # correlation_eval(model,data,n)
    # # kl_eval(model,data,n)
    # print(path)
    # folder_to_save = path.split('\\')[1][:-4]
    # path_to_save = f'figures/{folder_to_save}'
    # Path(path_to_save).mkdir(parents=True, exist_ok=True)
    # model.load_state_dict(torch.load(path,map_location=device))
    # real = torch.Tensor(data.trn.x)
    # dict_prints = {0: 'generated sample', 1: 'real sample'}
    # smp = model.model.sample(n, device)
    # calc_corr_cov(smp, data, path_to_save)
    # for feature in range(d):
    #     plt.figure()
    #     for i,example in enumerate([smp,torch.Tensor(real)]):
    #         kde = gaussian_kde(example[:,feature]).pdf(vv)
    #         plt.plot(vv, kde,'.', label=f'pdf {dict_prints[i]}')
    #     plt.xlabel('feature value')
    #     plt.ylabel('pdf')
    #     plt.title(f'pdf comparison NITS (generated {n} samples) vs train data for feature no. {feature}')
    #     plt.legend()
    # plt.show()
    # path_to_save_pdf = f'{path_to_save}/pdf_feature_{feature}.png'
    # plt.savefig(path_to_save_pdf)

    # print number of parameters
    print('number of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    print_every = 1
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

    time_ = time.time()
    epoch = 0
    train_ll = 0.
    max_val_ll = -np.inf
    patience = args.patience
    keep_training = True
    start_time = time.time()
    while keep_training:
        print('epoch', epoch, 'time [min]', round((time.time() - start_time) / 60), 'lr', optim.param_groups[0]['lr'])
        model.train()
        for i, (gamma, rad, x) in enumerate(
                create_dataloader(data.trn.gamma, data.trn.radiation, data.trn.x, batch_size=args.batch_size,device=device)):
            ll = model(x, (gamma, rad))
            optim.zero_grad()
            (-ll).backward()
            train_ll += ll.detach().cpu().numpy()
            optim.step()
            model.update()
        epoch += 1
        print('current ll loss:', ll / len(x))
        if epoch % print_every == 0:
            # compute train loss
            train_ll /= len(data.trn.x) * print_every
            lr = optim.param_groups[0]['lr']

            with torch.no_grad():
                model.eval()
                val_ll = 0.
                ema_val_ll = 0.
                for i, (gamma, rad, x) in enumerate(
                        create_dataloader(data.val.gamma, data.val.radiation, data.val.x, batch_size=args.batch_size, device=device)):
                    val_ll += model.model(x,(gamma,rad)).detach().cpu().numpy()
                    ema_val_ll += model(x,(gamma,rad)).detach().cpu().numpy()

                val_ll /= len(data.val.x)
                ema_val_ll /= len(data.val.x)

            # early stopping
            if ema_val_ll > max_val_ll + 1e-4:
                patience = args.patience
                max_val_ll = ema_val_ll
            else:
                patience -= 1
            print('Patience = ', patience)
            if patience <= np.ceil(args.patience / 2):
                scheduler.step()
            if patience == 0:
                print("Patience reached zero. max_val_ll stayed at {:.3f} for {:d} iterations.".format(max_val_ll,
                                                                                                       args.patience))
                max_vals_ll.append(max_val_ll)
                lasts_train_ll.append(train_ll)
                keep_training = False

            with torch.no_grad():
                model.eval()
                test_ll = 0.
                ema_test_ll = 0.
                for i, (gamma, rad, x) in enumerate(
                        create_dataloader(data.tst.gamma, data.tst.radiation, data.tst.x, batch_size=args.batch_size, device=device)):
                    test_ll += model.model(x,(gamma,rad)).detach().cpu().numpy()
                    ema_test_ll += model(x,(gamma,rad)).detach().cpu().numpy()

                test_ll /= len(data.tst.x)
                ema_test_ll /= len(data.tst.x)

            fmt_str1 = 'epoch: {:3d}, time: {:3d}s, train_ll: {:.4f},'
            fmt_str2 = ' ema_val_ll: {:.4f}, ema_test_ll: {:.4f},'
            fmt_str3 = ' val_ll: {:.4f}, test_ll: {:.4f}, lr: {:.2e}'

            print((fmt_str1 + fmt_str2 + fmt_str3).format(
                epoch,
                int(time.time() - time_),
                train_ll,
                ema_val_ll,
                ema_test_ll,
                val_ll,
                test_ll,
                lr))

            time_ = time.time()
            train_ll = 0.

        if epoch % (print_every * 10) == 0:
            print(args)
dict_to_print = {model_extra_string: (np.round(max_val, 3), np.round(last_train, 3)) for
                 model_extra_string, max_val, last_train in zip(model_names, max_vals_ll, lasts_train_ll)}
min_idx_val = np.argmin(max_vals_ll)
min_idx_train = np.argmin(lasts_train_ll)
print(f'best model according to val set: {model_names[min_idx_val]}')
print(f'best model according to train set: {model_names[min_idx_train]}')
print('dict_to_print:', dict_to_print)
# torch.save(model.state_dict(), f'models\\ANT_model_{model_extra_string}.pth')
