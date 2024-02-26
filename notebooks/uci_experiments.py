import argparse
import sys
sys.path.append(r'C:\Users\moshey\PycharmProjects\NITS')
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
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
        yield torch.tensor(x[idx:idx+batch_size], device=device)
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device)


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='miniboone')
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-b', '--batch_size', type=int, default=500)
parser.add_argument('-hi', '--hidden_dim', type=int, default=512)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=8)
parser.add_argument('-n', '--patience', type=int, default=6)
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
parser.add_argument('--bounds',type=list_str_to_list,default='[-3,3]')
args = parser.parse_args()
polyak_decay = [0.995]
for pd in polyak_decay:
    model_extra_string = f'polyak_decay_{pd}_{args.nits_arch}_bounds{args.bounds}'
    print(model_extra_string)
    args.polyak_decay = pd

    step_weights = np.array(args.step_weights)
    step_weights = step_weights / (np.sum(step_weights) + 1e-7)

    device = 'cuda:' + args.gpu if args.gpu else 'cpu'

    use_batch_norm = False
    zero_initialization = True
    weight_norm = False
    default_patience = 10
    if args.dataset == 'gas':
        # training set size: 852,174
        data = gas.GAS()
        default_dropout = 0.1
    elif args.dataset == 'power':
        # training set size: 1,659,917
        data = power.POWER()
        default_dropout = 0.4
    elif args.dataset == 'miniboone':
        # training set size: 29,556
        data = miniboone.MINIBOONE()
        default_dropout = 0.1
        args.hidden_dim = 128
        args.batch_size = 128
    elif args.dataset == 'hepmass':
        # training set size: 315,123
        data = hepmass.HEPMASS()
        default_dropout = 0.3
        default_patience = 3
        args.hidden_dim = 512
        args.batch_size = 1024
    elif args.dataset == 'bsds300':
        # training set size: 1,000,000
        data = bsds300.BSDS300()
        default_dropout = 0.2

    args.patience = args.patience if args.patience >= 0 else default_patience
    args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
    print(args)

    d = data.trn.x.shape[1]

    max_val = args.bounds[1] #max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
    min_val = args.bounds[0]#min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
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

    model = ResMADEModel(
        d=d,
        rotate=args.rotate,
        nits_model=nits_model,
        n_residual_blocks=args.n_residual_blocks,
        hidden_dim=args.hidden_dim,
        dropout_probability=args.dropout,
        use_batch_norm=use_batch_norm,
        zero_initialization=zero_initialization,
        weight_norm=weight_norm,
        nits_input_dim=nits_input_dim).to(device)

    shadow = ResMADEModel(
                d=d,
                rotate=args.rotate,
                nits_model=nits_model,
                n_residual_blocks=args.n_residual_blocks,
                hidden_dim=args.hidden_dim,
                dropout_probability=args.dropout,
                use_batch_norm=use_batch_norm,
                zero_initialization=zero_initialization,
                weight_norm=weight_norm,
                nits_input_dim=nits_input_dim
            ).to(device)

    # initialize weight norm
    if weight_norm:
        with torch.no_grad():
            for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
                params = model(x)
                break

    model = EMA(model, shadow, decay=args.polyak_decay).to(device)
    model.load_state_dict(torch.load(r"models/model_polyak_decay_0.995_[16, 16, 1]_bounds[-10, 10].pth",map_location=device))
    vv = np.linspace(-4, 4, num=7000)
    first_time = time.time()
    n = 2000
    smp = model.model.sample(n, device)
    time__ = time.time()
    print(f'time for sampling {n} samples: {time__ - first_time}')
    real = torch.Tensor(data.trn.x)
    dict_prints = {0: 'generated sample', 1: 'real sample'}
    for feature in range(d):
        plt.figure()
        for i,example in enumerate([smp,torch.Tensor(real)]):
            print(dict_prints[i])
            # params = model.model.mlp(example)
            # cdf_ = model.model.nits_model.cdf(example, params).detach().cpu().numpy()
            kde = gaussian_kde(example[:,feature]).pdf(vv)
            plt.plot(vv, kde,'.', label=f'pdf {dict_prints[i]}')
        #save figures to figures folder
        plt.title(f'pdf comparison NITS (generated {n} samples) vs train data for feature no. {feature}')
        plt.legend()
        plt.savefig(f'figures/arch_16_16_1_bounds10/feature_{feature}_pdf_comparison.png')


    # print number of parameters
    print('number of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    print_every = 10 if args.dataset != 'miniboone' else 1
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

    time_ = time.time()
    epoch = 0
    train_ll = 0.
    max_val_ll = -np.inf
    loss_op = lambda real, params: cnn_nits_loss(real, params, nits_model, discretized=args.discretized, dequantize=args.dequantize)
    patience = args.patience
    keep_training = True
    start_time = time.time()
    while keep_training:
        print('epoch', epoch, 'time [min]', round((time.time() - start_time)/60) , 'lr', optim.param_groups[0]['lr'])
        model.train()
        for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
            ll = model(x)
            ll_loss= -ll.mean()
            optim.zero_grad()
            ll_loss.backward()
            train_ll += ll.mean().detach().cpu().numpy()
            optim.step()
            model.update()
        print(f'current log-likelihood loss: {ll_loss.item()}')
        epoch += 1

        if epoch % print_every == 0:
            # compute train loss
            train_ll /= len(data.trn.x) * print_every
            lr = optim.param_groups[0]['lr']

            with torch.no_grad():
                model.eval()
                val_ll = 0.
                ema_val_ll = 0.
                for i, x in enumerate(create_batcher(data.val.x, batch_size=args.batch_size)):
                    x = torch.tensor(x, device=device)
                    val_ll += model.model(x).mean().detach().cpu().numpy()
                    ema_val_ll += model(x).mean().detach().cpu().numpy()

                val_ll /= len(data.val.x)
                ema_val_ll /= len(data.val.x)

            # early stopping
            if ema_val_ll > max_val_ll + 1e-4:
                patience = args.patience
                max_val_ll = ema_val_ll
            else:
                patience -= 1
            print('Patience = ',patience)
            if patience <= np.ceil(args.patience/2):
                scheduler.step()
            if patience == 0:
                print("Patience reached zero. max_val_ll stayed at {:.3f} for {:d} iterations.".format(max_val_ll, args.patience))
                keep_training = False

            with torch.no_grad():
                model.eval()
                test_ll = 0.
                ema_test_ll = 0.
                for i, x in enumerate(create_batcher(data.tst.x, batch_size=args.batch_size)):
                    x = torch.tensor(x, device=device)
                    test_ll += model.model(x).mean().detach().cpu().numpy()
                    ema_test_ll += model(x).mean().detach().cpu().numpy()

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
        torch.save(model.state_dict(), f'model_{model_extra_string}.pth')
