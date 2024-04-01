import torch
import numpy as np
def create_dataloader(gamma, radiation, params_scaled, batch_size, device,inv_or_forw = 'inverse'):
    gamma = downsample_gamma(gamma, 4)
    gamma = torch.tensor(gamma).to(device).float()
    radiation = torch.tensor(radiation).to(device).float()
    params_scaled = torch.tensor(params_scaled).to(device).float()
    if inv_or_forw == 'inverse':
        dataset = torch.utils.data.TensorDataset(gamma, radiation, params_scaled)
    elif inv_or_forw == 'forward_gamma':
        dataset = torch.utils.data.TensorDataset(params_scaled, gamma)
    elif inv_or_forw == 'forward_radiation':
        dataset = torch.utils.data.TensorDataset(params_scaled, downsample_radiation(radiation,rates=[4,2]))
    elif inv_or_forw == 'inverse_forward_gamma' or inv_or_forw == 'inverse_forward_GammaRad':
        dataset = torch.utils.data.TensorDataset(gamma, radiation)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

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

def downsample_gamma(gamma,rate):
    gamma_len = gamma.shape[1]
    gamma_mag = gamma[:,:int(gamma_len/2)]
    gamma_phase = gamma[:,int(gamma_len/2):]
    gamma_mag_downsampled = gamma_mag[:,::rate]
    gamma_phase_downsampled = gamma_phase[:,::rate]
    gamma_downsampled = np.concatenate((gamma_mag_downsampled,gamma_phase_downsampled),axis=1)
    return gamma_downsampled