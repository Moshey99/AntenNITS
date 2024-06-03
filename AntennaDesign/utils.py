import copy
import cv2
import scipy.io as sio
from scipy.ndimage import zoom
from os import listdir
import os
from pathlib import Path
from os.path import isfile, join
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import time
import trainer
import losses
from models import baseline_regressor, inverse_hypernet
import random
import pytorch_msssim
import glob
import pickle
import re
import open3d as o3d


class DataPreprocessor:
    def __init__(self):
        self.num_data_points = 10000
        # self.geometry_preprocessor()
        # self.gamma_preprocessor()
        # self.radiation_preprocessor()

    def radiation_preprocessor(self):
        all_radiations = []
        folder_path = 'C:\\Users\\moshey\\PycharmProjects\\etof_folder_git\\AntennaDesign_data\\data_10000x1\\data\\spectrum_moshe'
        for i in range(self.num_data_points):
            print('working on antenna number:', i + 1, 'out of:', self.num_data_points)
            im_resized = np.zeros((4, 181, 91))
            file_path = os.path.join(folder_path, f'ant{i}_farfield.txt')
            df = pd.read_csv(file_path, sep='\s+', skiprows=[0, 1], header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            arr = np.asarray(df)
            angle_res = arr[1, 0] - arr[0, 0]
            angle1_res = int(180 / angle_res + 1)
            angle2_res = int(360 / angle_res)
            im = arr[:, 3:7]
            im_resh = np.transpose(im.reshape(angle2_res, angle1_res, -1), (2, 0, 1))
            im_resh = im_resh[[0, 2, 1, 3], :, :]  # rearrange the channels to be [mag1, mag2, phase1, phase2]
            for j in range(im_resh.shape[0]):
                titles = ['mag1', 'mag2', 'phase1', 'phase2']
                plt.subplot(2, 2, j + 1)
                current = im_resh[j]
                im_resized[j] = np.clip(cv2.resize(current, (91, 181), interpolation=cv2.INTER_LINEAR), current.min(),
                                        current.max())
                if j < 2:
                    im_resized[j] = 10 * np.log10(im_resized[j])
                else:
                    im_resized[j] = np.deg2rad(im_resized[j]) - np.pi
                plt.imshow(im_resized[j])
                plt.title(titles[j])
                plt.colorbar()
            plt.show()
            all_radiations.append(im_resized)
        all_radiations = np.array(all_radiations)

        radiations_mag, radiations_phase = all_radiations[:, :2], all_radiations[:, 2:]
        radiations_phase_radians = np.deg2rad(radiations_phase) - np.pi
        radiations_mag_dB = 10 * np.log10(radiations_mag)
        radiations = np.concatenate((radiations_mag_dB, radiations_phase_radians), axis=1)
        saving_folder = os.path.join(Path(folder_path).parent, 'processed_data')
        np.save(os.path.join(saving_folder, 'radiations.npy'), radiations)
        print('Radiations saved successfully with mag in dB and phase in radians')

    def gamma_preprocessor(self):

        all_gammas = []
        folder_path = 'C:\\Users\\moshey\\PycharmProjects\\etof_folder_git\\AntennaDesign_data\\data_10000x1\\data\\spectrum_moshe'
        for i in range(self.num_data_points):
            print('working on antenna number:', i + 1, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, f'ant{i}_S11.pickle')
            with open(file_path, 'rb') as f:
                gamma_raw = pickle.load(f)
                gamma_complex = gamma_raw[0]
                gamma_mag, gamma_phase = np.abs(gamma_complex), np.angle(
                    gamma_complex)  # gamma_phase in radians already
                gamma_mag_dB = 10 * np.log10(gamma_mag)
                gamma = np.concatenate((gamma_mag_dB, gamma_phase))
                all_gammas.append(gamma)
        all_gammas = np.array(all_gammas)
        np.save(os.path.join(Path(folder_path).parent, 'processed_data', 'gammas.npy'), all_gammas)
        np.save(os.path.join(Path(folder_path).parent, 'processed_data', 'frequencies.npy'), gamma_raw[1])
        print('Gammas saved successfully with mag in dB and phase in radians')
        pass

    def geometry_preprocessor(self):
        voxel_size = 0.125
        min_bound_org = np.array([5.4, 3.825, 6.375]) - voxel_size
        max_bound_org = np.array([54., 3.83, 42.5]) + voxel_size
        bad_indices = [3234]
        for i in range(10000):
            if i==3234:
                continue
            mesh = o3d.io.read_triangle_mesh(
                rf"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_10000x1\data\models\{i}\Antenna_PEC_STEP.stl")
            vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=voxel_size,
                                                                                           min_bound=min_bound_org,
                                                                                           max_bound=max_bound_org)
            voxels = vg.get_voxels()
            indices = np.stack(list(vx.grid_index for vx in voxels))
            quary_x = np.arange(min_bound_org[0]+0.5*voxel_size,max_bound_org[0],step=voxel_size)
            quary_y = [3.825000047683716]
            quary_z = np.arange(min_bound_org[2]+0.5*voxel_size,max_bound_org[2],step=voxel_size)
            quary_array = np.zeros((len(quary_x), 1, len(quary_z)))
            start = time.time()
            for ii,x_val in enumerate(quary_x):
                for jj,y_val in enumerate(quary_y):
                    for kk,z_val in enumerate(quary_z):
                        ind = vg.get_voxel([x_val,y_val,z_val])
                        exists = np.any(np.all(indices == ind, axis=1))
                        quary_array[ii,jj,kk] = exists
            np.save(os.path.join(r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_10000x1\data\processed_data\voxels', f'voxels_{i}.npy'), quary_array)
            print(f'saved antenna {i}. Process time was:', time.time() - start)

    def stp_to_stl(self):
        import trimesh
        model_folder = 'C:\\Users\\moshey\\PycharmProjects\\etof_folder_git\\AntennaDesign_data\\data_10000x1\\data\\models'
        all =[]
        for i in range(10000):
            print('working on antenna number:', i + 1, 'out of:', self.num_data_points)
            stp_path = os.path.join(model_folder, f'{i}', 'Antenna_PEC_STEP.stp')
            mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(
                file_name=stp_path, gmsh_args=[("Mesh.Algorithm", 1), ("Mesh.CharacteristicLengthFromCurvature", 1),
                                               ("General.NumThreads", 10),
                                               ("Mesh.MinimumCirclePoints", 32)]))
            mesh.export(os.path.join(model_folder, f'{i}', 'Antenna_PEC_STEP.stl'))
        print('Geometries saved successfully as an stl formated 3D triangle mesh')


def create_dataloader(gamma, radiation, params_scaled, batch_size, device, inv_or_forw='inverse'):
    gamma = torch.tensor(gamma).to(device).float()
    radiation = torch.tensor(radiation).to(device).float()
    params_scaled = torch.tensor(params_scaled).to(device).float()
    if inv_or_forw == 'inverse':
        dataset = torch.utils.data.TensorDataset(gamma, radiation, params_scaled)
    elif inv_or_forw == 'forward_gamma':
        dataset = torch.utils.data.TensorDataset(params_scaled, gamma)
    elif inv_or_forw == 'forward_radiation':
        dataset = torch.utils.data.TensorDataset(params_scaled, downsample_radiation(radiation, rates=[4, 2]))
    elif inv_or_forw == 'inverse_forward_gamma' or inv_or_forw == 'inverse_forward_GammaRad':
        dataset = torch.utils.data.TensorDataset(gamma, radiation)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def nearest_neighbor_loss(loss_fn, x_train, y_train, x_val, y_val, k=1):
    strt = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_train)
    distances, indices = nbrs.kneighbors(x_val)
    cnt = len(np.where(distances < 0.1)[0])
    nearest_neighbor_y = y_train[indices].squeeze()
    loss = loss_fn(torch.tensor(nearest_neighbor_y), torch.tensor(y_val))
    return loss


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def split_dataset(dataset_path, train_val_test_split):
    folders = listdir(dataset_path)
    dataset_path_list = []
    folders = [folder for folder in folders if folder[:4] == 'wifi']
    for folder in folders:
        folder_path = join(dataset_path, folder)
        files = listdir(folder_path)
        for file in files:
            file_path = join(folder_path, file)
            if isfile(file_path):
                if file.endswith('.mat') and file.__contains__('results'):
                    dataset_path_list.append(file_path)

    dataset_path_list = np.array(dataset_path_list)
    num_of_data_points = len(dataset_path_list)
    num_of_train_points = int(num_of_data_points * train_val_test_split[0])
    num_of_val_points = int(num_of_data_points * train_val_test_split[1])

    train_pick = np.random.choice(num_of_data_points, num_of_train_points, replace=False)
    val_pick = np.random.choice(np.setdiff1d(np.arange(num_of_data_points), train_pick), num_of_val_points,
                                replace=False)
    if train_val_test_split[2] > 0:
        test_pick = np.setdiff1d(np.arange(num_of_data_points), np.concatenate((train_pick, val_pick)),
                                 assume_unique=True)
    else:
        test_pick = val_pick

    train_dataset_path_list = dataset_path_list[train_pick]
    val_dataset_path_list = dataset_path_list[val_pick]
    test_dataset_path_list = dataset_path_list[test_pick]
    return train_dataset_path_list, val_dataset_path_list, test_dataset_path_list


def create_dataset(dataset_path=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data',
                   train_val_test_split=[0.8, 0.1, 0.1]):
    dataset_path_list_train, dataset_path_list_val, dataset_path_list_test = split_dataset(dataset_path,
                                                                                           train_val_test_split)
    print('Creating dataset...')
    data_parameters_train, data_parameters_val, data_parameters_test = [], [], []
    data_gamma_train, data_gamma_val, data_gamma_test = [], [], []
    data_radiation_train, data_radiation_val, data_radiation_test = [], [], []

    for path in dataset_path_list_train:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_train.append(rad_concat_swapped)
        data_parameters_train.append(parameters)
        data_gamma_train.append(gamma)

    for path in dataset_path_list_val:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_val.append(rad_concat_swapped)
        data_parameters_val.append(parameters)
        data_gamma_val.append(gamma)

    for path in dataset_path_list_test:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_test.append(rad_concat_swapped)
        data_parameters_test.append(parameters)
        data_gamma_test.append(gamma)

    np.savez(dataset_path + '\\newdata.npz', parameters_train=np.array(data_parameters_train),
             gamma_train=np.array(data_gamma_train),
             radiation_train=np.array(data_radiation_train), parameters_val=np.array(data_parameters_val),
             gamma_val=np.array(data_gamma_val), radiation_val=np.array(data_radiation_val),
             parameters_test=np.array(data_parameters_test),
             gamma_test=np.array(data_gamma_test), radiation_test=np.array(data_radiation_test))
    print(f'Dataset created seccessfully. Saved in newdata.npz')


class standard_scaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def forward(self, input):
        return (input - self.mean) / self.std

    def inverse(self, input):
        return input * self.std + self.mean


def display_gamma(gamma):
    gamma_real = gamma[:int(gamma.shape[0] / 2)]
    gamma_imag = gamma[int(gamma.shape[0] / 2):]
    gamma_mag = np.sqrt(gamma_real ** 2 + gamma_imag ** 2)
    gamma_phase = np.arctan2(gamma_imag, gamma_real)
    plt.figure()
    plt.plot(np.arange(len(gamma_mag)), gamma_mag)
    plt.show()


def display_losses(train_loss, val_loss):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
    plt.legend()
    plt.show()


def display_gradients_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def downsample_gamma(gamma, rate=4):
    gamma_len = gamma.shape[1]
    gamma_mag = gamma[:, :int(gamma_len / 2)]
    gamma_phase = gamma[:, int(gamma_len / 2):]
    gamma_mag_downsampled = gamma_mag[:, ::rate]
    gamma_phase_downsampled = gamma_phase[:, ::rate]
    gamma_downsampled = np.concatenate((gamma_mag_downsampled, gamma_phase_downsampled), axis=1)
    return gamma_downsampled


def downsample_radiation(radiation, rates=[4, 2]):
    first_dim_rate, second_dim_rate = rates
    radiation_downsampled = radiation[:, :, ::first_dim_rate, ::second_dim_rate]
    return radiation_downsampled


def normalize_radiation(radiation, rad_range=[-55, 5]):
    min_val, max_val = rad_range[0], rad_range[1]
    radiation_mag = radiation[:, :int(radiation.shape[1] / 2)]
    radiation_mag = np.clip(radiation_mag, min_val, max_val)
    radiation_mag = (radiation_mag - min_val) / (max_val - min_val)
    radiation[:, :int(radiation.shape[1] / 2)] = radiation_mag
    return radiation


def convert_dataset_to_dB(data):
    print('Converting dataset to dB')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    train_gamma[:, :int(train_gamma.shape[1] / 2)] = 10 * np.log10(train_gamma[:, :int(train_gamma.shape[1] / 2)])
    val_gamma[:, :int(val_gamma.shape[1] / 2)] = 10 * np.log10(val_gamma[:, :int(val_gamma.shape[1] / 2)])
    test_gamma[:, :int(test_gamma.shape[1] / 2)] = 10 * np.log10(test_gamma[:, :int(test_gamma.shape[1] / 2)])
    train_radiation[:, :int(train_radiation.shape[1] / 2)] = 10 * np.log10(
        train_radiation[:, :int(train_radiation.shape[1] / 2)])
    val_radiation[:, :int(val_radiation.shape[1] / 2)] = 10 * np.log10(
        val_radiation[:, :int(val_radiation.shape[1] / 2)])
    test_radiation[:, :int(test_radiation.shape[1] / 2)] = 10 * np.log10(
        test_radiation[:, :int(test_radiation.shape[1] / 2)])
    np.savez(r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\newdata_dB.npz',
             parameters_train=train_params, gamma_train=train_gamma, radiation_train=train_radiation,
             parameters_val=val_params, gamma_val=val_gamma, radiation_val=val_radiation,
             parameters_test=test_params, gamma_test=test_gamma, radiation_test=test_radiation)
    print('Dataset converted to dB. Saved in data_dB.npz')


def reorganize_data(data):
    features_to_exclude = [5, 6]
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    all_params, all_gamma, all_radiation = np.concatenate((train_params, val_params, test_params),
                                                          axis=0), np.concatenate((train_gamma, val_gamma, test_gamma),
                                                                                  axis=0), np.concatenate(
        (train_radiation, val_radiation, test_radiation), axis=0)
    first_feature, second_feature = all_params[:, features_to_exclude[0]], all_params[:, features_to_exclude[1]]
    pct20first, pct30first = np.percentile(first_feature, 20), np.percentile(first_feature, 30)
    pct70second, pct80second = np.percentile(second_feature, 60), np.percentile(second_feature, 80)
    test_params_idx = np.where(np.logical_or(np.logical_and(first_feature > pct20first, first_feature < pct30first),
                                             np.logical_and(second_feature > pct70second,
                                                            second_feature < pct80second)))
    test_params_new, test_gamma_new, test_radiation_new = all_params[test_params_idx], all_gamma[test_params_idx], \
        all_radiation[test_params_idx]
    train_params_new, train_gamma_new, train_radiation_new = np.delete(all_params, test_params_idx, axis=0), np.delete(
        all_gamma, test_params_idx, axis=0), np.delete(all_radiation, test_params_idx, axis=0)
    val_idx = np.random.choice(train_params_new.shape[0], int(0.25 * train_params_new.shape[0]),
                               replace=False)  # 25% of remaining data is about 20% of original data
    val_params_new, val_gamma_new, val_radiation_new = train_params_new[val_idx], train_gamma_new[val_idx], \
        train_radiation_new[val_idx]
    train_params_new, train_gamma_new, train_radiation_new = np.delete(train_params_new, val_idx, axis=0), np.delete(
        train_gamma_new, val_idx, axis=0), np.delete(train_radiation_new, val_idx, axis=0)
    np.savez('data_reorganized.npz', parameters_train=train_params_new, gamma_train=train_gamma_new,
             radiation_train=train_radiation_new,
             parameters_val=val_params_new, gamma_val=val_gamma_new, radiation_val=val_radiation_new,
             parameters_test=test_params_new, gamma_test=test_gamma_new, radiation_test=test_radiation_new)


def produce_stats_gamma(GT_gamma, predicted_gamma, dataset_type='linear', to_print=True):
    if dataset_type == 'linear':
        GT_gamma[:, :int(GT_gamma.shape[1] / 2)] = 10 * np.log10(GT_gamma[:, :int(GT_gamma.shape[1] / 2)])
    if dataset_type == 'dB':
        pass
    if type(predicted_gamma) == tuple:
        GT_gamma, _ = GT_gamma
        predicted_gamma, _ = predicted_gamma
    predicted_gamma_mag, GT_gamma_mag = predicted_gamma[:, :int(predicted_gamma.shape[1] / 2)], GT_gamma[:, :int(
        GT_gamma.shape[1] / 2)]
    predicted_gamma_phase, GT_gamma_phase = predicted_gamma[:, int(predicted_gamma.shape[1] / 2):], GT_gamma[:, int(
        GT_gamma.shape[1] / 2):]
    # predicted_gamma_mag = 10*np.log10(predicted_gamma_mag)
    diff_dB = torch.abs(predicted_gamma_mag - GT_gamma_mag)
    respective_diff = torch.where(torch.abs(GT_gamma_mag) > 1.5, torch.div(diff_dB, torch.abs(GT_gamma_mag)) * 100, 0)
    avg_respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    avg_diff = torch.mean(diff_dB, dim=1)
    max_diff = torch.max(diff_dB, dim=1)[0]
    avg_max_diff = torch.mean(max_diff).item()
    diff_phase = predicted_gamma_phase - GT_gamma_phase
    while len(torch.where(diff_phase > np.pi)[0]) > 0 or len(torch.where(diff_phase < -np.pi)[0]) > 0:
        diff_phase[torch.where(diff_phase > np.pi)] -= 2 * np.pi
        diff_phase[torch.where(diff_phase < -np.pi)] += 2 * np.pi
    diff_phase = torch.abs(diff_phase)
    avg_diff_phase = torch.mean(diff_phase, dim=1)
    max_diff_phase = torch.max(diff_phase, dim=1)[0]
    avg_max_diff_phase = torch.mean(max_diff_phase).item()
    if to_print:
        print('gamma- ' + dataset_type + ' dataset - Avg diff: {:.4f} dB, Avg dB respective diff: {:.4f} % ,'
                                         ' Avg max diff: {:.4f} dB, Avg diff phase: {:.4f} rad, Avg max diff phase: {:.4f} rad'
              .format(torch.mean(avg_diff).item(), avg_respective_diff, avg_max_diff, torch.mean(avg_diff_phase).item(),
                      avg_max_diff_phase))

    return avg_diff, max_diff, avg_diff_phase, max_diff_phase


def produce_radiation_stats(predicted_radiation, gt_radiation, to_print=True):
    if type(predicted_radiation) == tuple:
        _, predicted_radiation = predicted_radiation
        _, gt_radiation = gt_radiation
    sep = predicted_radiation.shape[1] // 2
    pred_rad_mag, gt_rad_mag = predicted_radiation[:, :sep], gt_radiation[:, :sep]
    pred_rad_phase, gt_rad_phase = predicted_radiation[:, sep:], gt_radiation[:, sep:]
    abs_diff_mag = torch.abs(pred_rad_mag - gt_rad_mag)
    respective_diff = torch.where(torch.abs(gt_rad_mag) > 1.5, torch.div(abs_diff_mag, torch.abs(gt_rad_mag)) * 100, 0)
    respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    diff_phase = pred_rad_phase - gt_rad_phase
    while len(torch.where(diff_phase > np.pi)[0]) > 0 or len(torch.where(diff_phase < -np.pi)[0]) > 0:
        diff_phase[torch.where(diff_phase > np.pi)] -= 2 * np.pi
        diff_phase[torch.where(diff_phase < -np.pi)] += 2 * np.pi
    max_diff_mag = torch.amax(abs_diff_mag, dim=(1, 2, 3))
    mean_abs_error_mag = torch.mean(torch.abs(abs_diff_mag), dim=(1, 2, 3))
    mean_max_error_mag = torch.mean(max_diff_mag).item()
    abs_diff_phase = torch.abs(diff_phase)
    max_diff_phase = torch.amax(abs_diff_phase, dim=(1, 2, 3))
    mean_abs_error_phase = torch.mean(abs_diff_phase, dim=(1, 2, 3))
    mean_max_error_phase = torch.mean(max_diff_phase).item()
    msssim_vals = []
    for i in range(gt_radiation.shape[0]):
        msssim_vals.append(pytorch_msssim.msssim(pred_rad_mag[i:i + 1].float(), gt_rad_mag[i:i + 1].float()).item())
    msssim_vals = torch.tensor(msssim_vals)
    avg_msssim_mag = msssim_vals.mean().item()
    # print all the stats for prnt variant as one print statement
    if to_print:
        print('Radiation - mean_abs_error_mag:', round(torch.mean(mean_abs_error_mag).item(), 4),
              ' dB, mean dB respective error: ', round(respective_diff, 4)
              , '%, mean_max_error_mag:', round(mean_max_error_mag, 4)
              , ' dB, mean_abs_error_phase:', round(torch.mean(mean_abs_error_phase).item(), 4),
              ' rad, mean_max_error_phase:'
              , round(mean_max_error_phase, 4), ' rad, msssim_mag:', round(avg_msssim_mag, 4))
    return mean_abs_error_mag, max_diff_mag, msssim_vals


def save_antenna_mat(antenna: torch.Tensor, path: str, scaler: standard_scaler):
    import scipy.io as sio
    antenna = antenna.detach().cpu().numpy()
    antenna_unscaled = scaler.inverse(antenna)
    sio.savemat(path, {'antenna': antenna_unscaled})


class DXF2IMG(object):
    default_img_format = '.png'
    default_img_res = 300

    def convert_dxf2img(self, names, img_format=default_img_format, img_res=default_img_res):
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
        for name in names:
            doc = ezdxf.readfile(name)
            msp = doc.modelspace()
            # Recommended: audit & repair DXF document before rendering
            auditor = doc.audit()
            # The auditor.errors attribute stores severe errors,
            # which *may* raise exceptions when rendering.
            if len(auditor.errors) != 0:
                raise exception("The DXF document is damaged and can't be converted!")
            else:
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                ctx = RenderContext(doc)
                ctx.set_current_layout(msp)
                ctx.current_layout_properties.set_colors(bg='#FFFFFF')
                out = MatplotlibBackend(ax)
                Frontend(ctx, out).draw_layout(msp, finalize=True)

                img_name = re.findall("(\S+)\.", name)  # select the image name that is the same as the dxf file name
                first_param = ''.join(img_name) + img_format  # concatenate list and string
                fig.savefig(first_param, dpi=img_res)


if __name__ == '__main__':
    data_processor = DataPreprocessor()
    print(1)
