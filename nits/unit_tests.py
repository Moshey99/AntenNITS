import torch
from nits.model import *
from nits.autograd_model import *

# device = 'cpu'
device = 'cuda:5'

n = 32
start, end = -2., 2.
arch = [1, 8, 8, 1]
monotonic_const = 1e-2

print("Testing NITS.")

for d in [1, 2, 10]:
    for constraint_type in ['neg_exp', 'exp']:
        for final_layer_constraint in ['softmax', 'exp']:
#             print("""
#             Testing configuration:
#                 d: {}
#                 constraint_type: {}
#                 final_layer_constraint: {}
#                   """.format(d, constraint_type, final_layer_constraint))
            ############################
            # DEFINE MODELS            #
            ############################

            model = NITS(d=d, start=start, end=end, arch=arch,
                                 monotonic_const=monotonic_const, constraint_type=constraint_type,
                                 final_layer_constraint=final_layer_constraint).to(device)
            params = torch.randn((n, d * model.n_params)).to(device)

            ############################
            # SANITY CHECKS            #
            ############################

            # check that the function integrates to 1
            assert torch.allclose(torch.ones((n, d)).to(device),
                                  model.cdf(model.end, params) - model.cdf(model.start, params), atol=1e-5)

            # check that the pdf is all positive
            z = torch.linspace(start, end, steps=n, device=device)[:,None].tile((1, d))
            assert (model.pdf(z, params) >= 0).all()

            # check that the cdf is the inverted
            cdf = model.cdf(z, params[0:1])
            icdf = model.icdf(cdf, params[0:1])
            assert (z - icdf <= 1e-3).all()

            ############################
            # COMPARE TO AUTOGRAD NITS #
            ############################
            autograd_model = ModelInverse(arch=arch, start=start, end=end, store_weights=False,
                                          constraint_type=constraint_type, monotonic_const=monotonic_const,
                                          final_layer_constraint=final_layer_constraint).to(device)

            def zs_params_to_forwards(zs, params):
                out = []
                for z, param in zip(zs, params):
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.apply_layers(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            autograd_outs = zs_params_to_forwards(z, params)
            outs = model.forward_(z, params)
            assert torch.allclose(autograd_outs, outs, atol=1e-4)

            def zs_params_to_cdfs(zs, params):
                out = []
                for z, param in zip(zs, params):
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.cdf(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            autograd_outs = zs_params_to_cdfs(z, params)
            outs = model.cdf(z, params)
            assert torch.allclose(autograd_outs, outs, atol=1e-4)

            def zs_params_to_pdfs(zs, params):
                out = []
                for z, param in zip(zs, params):
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.pdf(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            autograd_outs = zs_params_to_pdfs(z, params)
            outs = model.pdf(z, params)
            assert torch.allclose(autograd_outs, outs, atol=1e-4)

            def zs_params_to_icdfs(zs, params):
                out = []
                for z, param in zip(zs, params):
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.F_inv(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            y = torch.rand((n, d)).to(device)
            autograd_outs = zs_params_to_icdfs(y, params)
            outs = model.icdf(y, params)
            assert torch.allclose(autograd_outs, outs, atol=1e-1)

            # try with single parameter, many zs

            def zs_params_to_pdfs(zs, param):
                out = []
                for z in zs:
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.pdf(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            autograd_outs = zs_params_to_pdfs(z, params[0])
            outs = model.pdf(z, params[0:1])
            assert torch.allclose(autograd_outs, outs, atol=1e-4)

            # try with single z, many parameters

            def zs_params_to_pdfs(z, params):
                out = []
                for param in params:
                    for d_ in range(d):
                        start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
                        autograd_model.set_params(param[start_idx:end_idx])
                        out.append(autograd_model.pdf(z[d_:d_+1][None,:]))

                out = torch.cat(out, axis=0).reshape(-1, d)
                return out

            autograd_outs = zs_params_to_pdfs(z[0], params)
            outs = model.pdf(z[0:1], params)
            assert torch.allclose(autograd_outs, outs, atol=1e-4)

from nits.discretized_mol import *
print("Testing arch = [1, 10, 1], 'neg_exp' constraint_type, 'softmax' final_layer_constraint " \
      "against discretized mixture of logistics.")

model = NITS(d=1, start=-1e5, end=1e5, arch=[1, 10, 1],
                     monotonic_const=0., constraint_type='neg_exp',
                     final_layer_constraint='softmax').to(device)
params = torch.randn((n, model.n_params, 1, 1))
z = torch.randn((n, 1, 1, 1))

loss1 = discretized_mix_logistic_loss_1d(z, params)
loss2 = discretized_nits_loss(z, params, arch=[1, 10, 1], nits_model=model)

assert (loss1 - loss2).norm() < 1e-2, (loss1 - loss2).norm()

model = NITS(d=1, start=-1e7, end=1e7, arch=[1, 10, 1],
                     monotonic_const=0., constraint_type='neg_exp',
                     final_layer_constraint='softmax').to(device)

loss1 = discretized_mix_logistic_loss_1d(z, params)
loss2 = discretized_nits_loss(z, params, arch=[1, 10, 1], nits_model=model)

assert (loss1 - loss2).norm() < 1e-3, (loss1 - loss2).norm()

print("All tests passed!")

print("Testing Conditional NITS.")
start, end = -2., 2.
monotonic_const = 1e-2
d = 4
c_arch = [d, 8, 8, 1]
constraint_type = 'exp'
final_layer_constraint = 'softmax'
device = 'cpu'

c_model = ConditionalNITS(d=d, start=start, end=end, arch=c_arch,
                          monotonic_const=monotonic_const, constraint_type=constraint_type,
                          final_layer_constraint=final_layer_constraint,
                          autoregressive=False).to(device)

c_params = torch.randn((n, c_model.tot_params)).to(device)
z = torch.linspace(start, end, steps=n, device=device)[:,None].tile((1, d)).to(device)

def cond_zs_params_to_cdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])
            out.append(c_autograd_model.cdf(z[None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = cond_zs_params_to_cdfs(z, c_params)
outs = c_model.cdf(z, c_params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

def cond_zs_params_to_pdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])
            out.append(c_autograd_model.pdf(z[None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = cond_zs_params_to_pdfs(z, c_params)
outs = c_model.pdf(z, c_params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

# testing the inverse_cdf function

def cond_zs_params_to_icdfs(ys, zs, params):
    out = []
    for y, z, param in zip(ys, zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])
            out.append(c_autograd_model.F_inv(y[d_:d_+1][None,:], given_x=z[None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

y = torch.rand((n, d)).to(device)
autograd_outs = cond_zs_params_to_icdfs(y, z, c_params)
outs = c_model.icdf(y, c_params, given_x=z)
assert torch.allclose(autograd_outs, outs, atol=1e-1)

for i in range(d):
    tmp = torch.cat([z[:,:i], outs[:,i:i+1], z[:,i+1:]], axis=1)
    res = c_model.cdf(tmp, c_params)
    assert torch.allclose(res[:,i], y[:,i], atol=1e-2)

for i in range(d):
    tmp = torch.cat([z[:,:i], outs[:,i:i+1], z[:,i+1:]], axis=1)
    res = cond_zs_params_to_cdfs(tmp, c_params)
    assert torch.allclose(res[:,i], y[:,i], atol=1e-2)

print("All tests passed!")

print('Testing autoregressive conditional NITS.')
start, end = -2., 2.
monotonic_const = 1e-2
constraint_type = 'exp'
final_layer_constraint = 'softmax'
device = 'cpu'

c_model = ConditionalNITS(d=d, start=start, end=end, arch=c_arch,
                          monotonic_const=monotonic_const, constraint_type=constraint_type,
                          final_layer_constraint=final_layer_constraint,
                          autoregressive=True).to(device)

c_params = torch.randn((n, c_model.tot_params)).to(device)
z = torch.linspace(start, end, steps=n, device=device)[:,None].tile((1, d)).to(device)

def causal_mask(x, i):
    x = x.clone()[None,:]
    x[:,i+1:] = 0.
    return x

def cond_zs_params_to_cdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])

            # set mask and apply function
            z_masked = causal_mask(z, d_)
            out.append(c_autograd_model.cdf(z_masked))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = cond_zs_params_to_cdfs(z, c_params)
outs = c_model.cdf(z, c_params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

def cond_zs_params_to_pdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])

            # set mask and apply function
            z_masked = causal_mask(z, d_)
            out.append(c_autograd_model.pdf(z_masked))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = cond_zs_params_to_pdfs(z, c_params)
outs = c_model.pdf(z, c_params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

# testing the inverse_cdf function

def cond_zs_params_to_icdfs(ys, zs, params):
    out = []
    for y, z, param in zip(ys, zs, params):
        for d_ in range(d):
            c_autograd_model = ModelInverse(arch=c_arch, start=start, end=end, store_weights=False,
                                           constraint_type=constraint_type, monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint,
                                           non_conditional_dim=d_).to(device)
            start_idx, end_idx = d_ * c_autograd_model.n_params, (d_ + 1) * c_autograd_model.n_params
            c_autograd_model.set_params(param[start_idx:end_idx])

            # set mask and apply function
            z_masked = torch.cat(out[len(out)-d_:] + [torch.zeros((1, d - d_))], axis=1)
            out.append(c_autograd_model.F_inv(y[d_:d_+1][None,:], given_x=z_masked))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

y = torch.rand((n, d)).to(device)
autograd_outs = cond_zs_params_to_icdfs(y, z, c_params)
outs = c_model.icdf(y, c_params)
assert torch.allclose(autograd_outs, outs, atol=1e-1)

assert torch.allclose(c_model.cdf(outs, c_params), y, atol=1e-3)
assert torch.allclose(cond_zs_params_to_cdfs(autograd_outs, c_params), y, atol=1e-3)

print("All tests passed!")

print("Passed all unit tests!")
