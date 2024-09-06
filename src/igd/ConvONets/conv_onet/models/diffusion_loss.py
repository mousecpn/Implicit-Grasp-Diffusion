import torch
from igd.ConvONets.conv_onet.models.diffusion import SO3_R3
from theseus import SO3
import torch.nn as nn
import torch.nn.functional as F
import theseus as th
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput, randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from typing import List, Optional, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import time
from einops import rearrange, reduce
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def colorline(
    x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, linestyle=':')

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments






class posrotDiffusion(nn.Module):
    def __init__(self, schedulers="DDPM", condition_mask=[0,0,0,0,0,0,0]):
        super().__init__()
        if schedulers == "DDPM":
            self.rot_noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2')
            self.pos_noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='scaled_linear')
        self.num_inference_steps = 100
        condition_mask = torch.FloatTensor(condition_mask).bool()


        self.register_buffer('condition_mask', condition_mask)
    
    def loss_fn(self, model, model_input):
        data = model_input['x_ene_pos']
        if 'visual_context' in model_input.keys():
            c = model_input['visual_context']
        if 'width' in model_input.keys():
            width = model_input['width']
        if 'global_feature' in model_input.keys():
            global_feature = model_input['global_feature']
        else:
            global_feature = None
        batch_size, sample_num = data.shape[0], data.shape[1]

        if 'width' in model_input.keys():
            data = torch.cat((data, width.reshape(batch_size, sample_num//2, -1).repeat(1,2,1)), dim=-1)

        noise_pos = torch.randn(data[...,:3].shape, device=data.device)
        noise_rot = torch.randn(data[...,3:].shape, device=data.device)

        noise = torch.cat((noise_pos, noise_rot), dim=-1)


        timesteps = torch.randint(
            0, self.rot_noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=data.device
        ).long()

        noisy_data_pos = self.pos_noise_scheduler.add_noise(
            data[...,:3], noise_pos, timesteps)
        
        noisy_data_rot = self.rot_noise_scheduler.add_noise(
            data[...,3:], noise_rot, timesteps)
        
        noisy_data = torch.cat((noisy_data_pos, noisy_data_rot), dim=-1)
        
        # compute loss mask
        loss_mask = ~self.condition_mask

        # apply conditioning
        noisy_data[..., self.condition_mask] = data[..., self.condition_mask]
        if global_feature is not None:
            pred = model(noisy_data, c, timesteps, global_cond=global_feature)
        else:
            pred = model(noisy_data, c, timesteps)

        pred_type = self.rot_noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = data
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        # pred[...,-1] *= 40
        # target[...,-1] *= 40
        loss = F.mse_loss(pred, target, reduction='none')
        # loss *= 0.01
        loss = loss[..., loss_mask]
        loss_rot = loss[...,3:]
        loss_pos = loss[...,:3]
        # loss = loss * loss_mask.type(loss.dtype)
        loss_rot = reduce(loss_rot, 'b ... -> b (...)', 'mean')
        loss_pos = reduce(loss_pos, 'b ... -> b (...)', 'mean')
        loss_pos = loss_pos.mean(-1)
        loss_rot = loss_rot.mean(-1)
        return loss_pos, loss_rot

    def sample_data(self, HT, c, model):
        # batch_size, sample_num = shape[:2]
        batch_size = HT.size(0)
        sample_num = HT.size(1)
        # sample_num = c.size(1)
        # condition_mask = self.condition_mask

        rot_scheduler = self.rot_noise_scheduler
        pos_scheduler = self.pos_noise_scheduler
        data = torch.randn((batch_size, sample_num, 7), device=c.device)
        
        condition_data = data

        pos_scheduler.set_timesteps(self.num_inference_steps)

        for t in pos_scheduler.timesteps:
            # 1. apply conditioning
            data[..., self.condition_mask] = condition_data[..., self.condition_mask]

            # t = t[None].to(data.device)
            # t = t.expand(data.shape[0])

            # 2. predict model output
            model_output = model(data, c, t[None].to(data.device).expand(data.shape[0]))

            # 3. compute previous image: x_t -> x_t-1
            pos_data = pos_scheduler.step(
                model_output[...,:3], t, data[...,:3], 
                generator=None,
                ).prev_sample
            rot_data = rot_scheduler.step(
                model_output[...,3:], t, data[...,3:], 
                generator=None,
                ).prev_sample
            data = torch.cat((pos_data, rot_data), dim=-1).reshape(batch_size, sample_num, -1)
            # data = torch.cat((Ht[...,:3, -1].reshape(batch_size,sample_num,3), data), dim=-1)
        
        # finally make sure conditioning is enforced
        data[..., self.condition_mask] = condition_data[..., self.condition_mask]
        data[...,3:] = nn.functional.normalize(data[...,3:], dim=2)
        # width = data[...,-1:].reshape(batch_size,sample_num)
        return data
    
def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

def predict_start_from_noise(scheduler, x_t, noise, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
    timesteps = timesteps.to(x_t.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(x_t.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    original_samples = (x_t - sqrt_one_minus_alpha_prod * noise)/sqrt_alpha_prod
    
    return original_samples
    

class Diffusion(nn.Module):
    def __init__(self, schedulers="DDPM", condition_mask=[1,1,1,0,0,0,0], beta_schedule='squaredcos_cap_v2',prediction_type='sample', quality=False):
        super().__init__()
        self.schedulers = schedulers
        if schedulers == "DDPM":
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type)
            # self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=1e-7, beta_end=9e-3, beta_schedule='sigmoid', prediction_type=prediction_type)
        elif schedulers == "DEIS":
            self.noise_scheduler = DEISMultistepScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type, solver_order=2)
        elif schedulers == "DPM":
            self.noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type)
        self.num_inference_steps = 100
        condition_mask = torch.FloatTensor(condition_mask).bool()
        self.quality = quality
        self.viz_trajectory = []

        self.register_buffer('condition_mask', condition_mask)
    
    def noise_quaternion(self, mu):
        batch_size, sample_num = mu.shape[0], mu.shape[1]
        mu = mu.reshape(batch_size*sample_num, -1)
        z = torch.normal(0, 1, (batch_size*sample_num, 4)).to(mu.device)
        z = nn.functional.normalize(z, dim=-1)
        z = z[:,None,:] - (torch.bmm(z[:,None,:], mu[..., None])) * mu[:, None, :]
        noise = z.reshape(batch_size, -1, 4)
        return noise
    
    def loss_fn(self, model, model_input):
        grasp = model_input['x_ene_pos']
        c = model_input['visual_context']
        if 'width' in model_input.keys():
            width = model_input['width']
        if 'label' in model_input.keys():
            label = model_input['label']
        if 'global_feature' in model_input.keys():
            global_feature = model_input['global_feature']
        else:
            global_feature = None
        batch_size, sample_num = c.shape[0], c.shape[1]
   
        p = grasp[...,:3]
        xw = grasp[...,3:]
        if 'width' in model_input.keys():
            data = torch.cat((xw, width.reshape(batch_size, sample_num, -1)), dim=-1)
        else:
            data = torch.cat((p, xw), dim=-1)
        noise = torch.randn(data.shape, device=data.device)

        # if self.schedulers == "DEIS":
        #     num_inference_steps = 10
        #     # t_start = 8

        #     self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        #     timesteps = torch.randint(
        #         1, self.noise_scheduler.config.num_train_timesteps, 
        #         (batch_size,), device=data.device
        #     ).long()

        #     # add noise
        #     # timesteps = self.noise_scheduler.timesteps[t_start * self.noise_scheduler.order :]
        #     new_noise = add_noise(self.noise_scheduler, data, noise, timesteps)
        # else:
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=data.device
        ).long()
        

        noisy_data = self.noise_scheduler.add_noise(
            data, noise, timesteps)
        

        ################## re-noise mechnanism #############################
        # t1 = time.time()
        anchor = torch.linspace(0, sample_num-1, sample_num).unsqueeze(0).repeat(batch_size, 1).to(data.device)
        new_noise = noise.clone()
        while True:
            noise_data_ = noisy_data.clone()
            min_dist = (noise_data_[:,:,None,:] - grasp[:,None,:,:]).pow(2).sum(-1).pow(0.5) # (bs, ns, ns)
            idx = torch.argmin(min_dist, dim=-1)
            renoise_mask = (idx != anchor)
            if renoise_mask.sum()< 10:
                break
            # re-generate noise
            new_noise = torch.randn(data.shape, device=data.device)
            # new_noise = new_noise + (data-new_noise) * 0.1
            new_noisy_data = self.noise_scheduler.add_noise(
                data, new_noise, timesteps)
            noise[renoise_mask] = new_noise[renoise_mask]
            noisy_data[renoise_mask] = new_noisy_data[renoise_mask]
        # print("regenerate time:",time.time()-t1)

        
        # compute loss mask
        loss_mask = ~self.condition_mask

        # apply conditioning
        noisy_data[..., self.condition_mask] = data[..., self.condition_mask]
        if global_feature is not None:
            pred = model(noisy_data, c, timesteps, global_cond=global_feature)
        else:
            pred = model(noisy_data, c, timesteps)
        if self.quality:
            qual = pred[...,-1]
            pred = pred[...,:-1]

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = data
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        
        ############## quaternion loss reweighted ################
        # x_0 = predict_start_from_noise(self.noise_scheduler, noisy_data, pred, timesteps)
        # rot_target = data[...,loss_mask].reshape(batch_size, sample_num, -1)
        # rot_pred = x_0[...,loss_mask]
        # rot_pred = nn.functional.normalize(rot_pred, dim=-1).reshape(batch_size, sample_num, -1)

        # quaternion_loss = (1.0 - torch.abs(torch.sum(rot_pred * rot_target, dim=-1))).detach()
        # quaternion_loss[label.bool()] /= quaternion_loss[label.bool()].sum()
        
        # loss = F.mse_loss(pred, target, reduction='none')
        # # loss *= 0.01
        # loss = loss[..., loss_mask]
        # loss = quaternion_loss.unsqueeze(2)[label.bool()] * loss[label.bool()]
        # loss = loss.sum(-1)
        ############## quaternion loss reweighted ################
        
        # original loss
        # loss = loss * loss_mask.type(loss.dtype)
        if self.quality:
            label = torch.exp(-loss.mean(-1).detach())
            # loss_qual = F.binary_cross_entropy(torch.sigmoid(qual), label, reduction='none')
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss[..., loss_mask]
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean(-1)
        
        # quaternion loss
        # rot_target = target[...,loss_mask].reshape(batch_size*sample_num, -1)
        # rot_pred = pred[...,loss_mask]
        # rot_pred = nn.functional.normalize(rot_pred, dim=-1).reshape(batch_size*sample_num, -1)

        # loss = F.mse_loss(rot_pred, rot_target, reduction='none')
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean(-1)
        # loss = 1.0 - torch.abs(torch.sum(rot_pred * rot_target, dim=-1))
        if self.quality:
            return loss, qual, label
        return loss

    def sample_data(self, HT, c, model):
        batch_size = HT.size(0)
        sample_num = HT.size(1)
        # condition_mask = self.condition_mask

        scheduler = self.noise_scheduler
        
        Ht = HT

        H0_in = SO3_R3(R=Ht[...,:3,:3].reshape(-1,3,3), t=Ht[...,:3, -1].reshape(-1,3))
        noisy_quaternion = torch.randn((batch_size,sample_num,4), device=Ht.device)
        # viz_traj = []
        # viz_traj.append(noisy_quaternion[0,0,:])
        # noisy_quaternion = self.noise_quaternion(noisy_quaternion)
        data = torch.cat((Ht[...,:3, -1].reshape(batch_size,sample_num,3), noisy_quaternion), dim=-1)
        # xw = H0_in.log_map().reshape(batch_size, sample_num,-1)
        # xw = H0_in.log_map().reshape(batch_size, sample_num,-1)
        # width = torch.randn(
        #     size=(batch_size, sample_num,1), 
        #     dtype=xw.dtype,
        #     device=xw.device)
        # data = torch.cat((xw[...,3:], width), dim=-1)
        # data = xw[...,3:]
        condition_data = data

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            data[..., self.condition_mask] = condition_data[..., self.condition_mask]

            # t = t[None].to(data.device)
            # t = t.expand(data.shape[0])

            # 2. predict model output
            model_output = model(data, c, t[None].to(data.device).expand(data.shape[0]))
            if self.quality:
                qual = model_output[...,-1]
                model_output = model_output[...,:-1]
            
            # # obtain tangent space
            # model_output = model_output.reshape(batch_size*sample_num, -1)
            # model_output[:,~self.condition_mask] = nn.functional.normalize(model_output[:,~self.condition_mask], dim=-1)
            # model_output[:, None, ~self.condition_mask] = model_output[:, None, ~self.condition_mask] - (torch.bmm(model_output[:,None,~self.condition_mask], data[...,~self.condition_mask].reshape(-1,4)[..., None])) * data[...,~self.condition_mask].reshape(-1,4)[:, None, :]
            # model_output = model_output.reshape(batch_size, sample_num, -1)

            # 3. compute previous image: x_t -> x_t-1
            data = scheduler.step(
                model_output, t, data, 
                generator=None,
                ).prev_sample
            data = data.reshape(batch_size, sample_num, -1)
            # viz_traj.append(data[0,0,3:])
            # data = torch.cat((Ht[...,:3, -1].reshape(batch_size,sample_num,3), data), dim=-1)
        
        # finally make sure conditioning is enforced
        data[..., self.condition_mask] = condition_data[..., self.condition_mask]
        # self.viz_trajectory.append(torch.stack(viz_traj, dim=0).detach().cpu().numpy())

        # width = data[...,-1:].reshape(batch_size,sample_num)
        if self.quality:
            return data, qual
        return data
    
    def visualize_denoising_process(self, target, trajectory):
        """
        target: quaternion, (x,y,z,w) array(2, 4)
        trajectory: array(n, t, 4)
        """
        # center = (0, 0)
        # radius = 1
        fig, ax = plt.subplots()
        # circle = plt.Circle(center, radius, fill=False, label='Circle')
        # ax.add_patch(circle)
        n,t,_= trajectory.shape

        target = np.concatenate((target,-target),axis=0)

        # 设置坐标轴的范围
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)

        # 添加标签和标题
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title('Diffusion process')
        # select_channel
        all_data = np.concatenate((target.reshape(-1,4), trajectory.reshape(-1,4)),axis=0)

        all_data_tsne = TSNE().fit_transform(all_data)

        vis_target = all_data_tsne[:4, :]

        trajectory_tsne = all_data_tsne[4:,:].reshape(n,t,-1)
        plt.scatter(vis_target[:,0], vis_target[:,1], c='green', marker='*', s=400)

        for i in range(trajectory_tsne.shape[0]):
            denoise_data = trajectory_tsne[i]
            z = np.linspace(0, 1, len(denoise_data[:,0]))
            colorline(denoise_data[:,0], denoise_data[:,1], z, cmap=plt.get_cmap('jet'), linewidth=2)
            # plt.plot(denoise_data[:,0], denoise_data[:,1], color=custom_cmap(0.5), linestyle='--')
        plt.scatter(trajectory_tsne[:,-1,0], trajectory_tsne[:,-1,1], c='red', marker='^', s=100)
        plt.scatter(trajectory_tsne[:,0,0], trajectory_tsne[:,0,1], c='blue', marker='s', s=100)
        plt.show()
        return
        



    def guided_sample(self, HT, c, model, classifier):
        batch_size = HT.size(0)
        sample_num = HT.size(1)
        # condition_mask = self.condition_mask

        scheduler = self.noise_scheduler
        
        Ht = HT

        H0_in = SO3_R3(R=Ht[...,:3,:3].reshape(-1,3,3), t=Ht[...,:3, -1].reshape(-1,3))
        xw = H0_in.log_map().reshape(batch_size, sample_num,-1)
        # width = torch.randn(
        #     size=(batch_size, sample_num,1), 
        #     dtype=xw.dtype,
        #     device=xw.device)
        # data = torch.cat((xw[...,3:], width), dim=-1)
        data = xw[...,3:]
        condition_data = data

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            data[..., self.condition_mask] = condition_data[..., self.condition_mask]

            # t = t[None].to(data.device)
            # t = t.expand(data.shape[0])

            # 2. predict model output
            model_output = model(data, c, t[None].to(data.device).expand(data.shape[0]))
        
            # 3. compute previous image: x_t -> x_t-1
            data = scheduler.step(
                model_output, t, data, 
                generator=None,
                ).prev_sample
            data = data.reshape(batch_size, sample_num, -1)
            grasp_data = torch.cat((xw[...,3:], data), dim=-1)
            quals = classifier(grasp_data)
            label = torch.ones_like(quals)
            loss = F.binary_cross_entropy(quals, label)
            grad = torch.autograd.grad(loss.sum(), data,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]
            data -= grad
            
        
        # finally make sure conditioning is enforced
        data[..., self.condition_mask] = condition_data[..., self.condition_mask]

        # width = data[...,-1:].reshape(batch_size,sample_num)
        return data[...,:3]


class SE3DenoisingLoss():

    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def log_gaussian_on_lie_groups(self, x, context):
        R_p = SO3.exp_map(x[...,3:])
        delta_H = th.compose(th.inverse(context[0]), R_p)
        log = delta_H.log_map()

        dt = x[...,:3] - context[1]

        tlog = torch.cat((dt, log), -1)
        return -0.5 * tlog.pow(2).sum(-1)/(context[2]**2)

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## From Homogeneous transformation to axis-angle ##
        H = model_input['x_ene_pos']
        n_grasps = H.shape[1]
        c = model_input['visual_context']
        model.set_latent(c, batch=n_grasps)

        H_in = H.reshape(-1, 4, 4)
        H_in = SO3_R3(R=H_in[:, :3, :3], t=H_in[:, :3, -1])
        tw = H_in.log_map()
        #######################

        ## 1. Compute noisy sample SO(3) + R^3##
        random_t = torch.rand_like(tw[...,0], device=tw.device) * (1. - eps) + eps
        z = torch.randn_like(tw)
        std = self.marginal_prob_std(random_t)
        noise = z * std[..., None]
        noise_t = noise[..., :3]
        noise_rot = SO3.exp_map(noise[...,3:])
        R_p = th.compose(H_in.R, noise_rot)
        t_p = H_in.t + noise_t
        #############################

        ## 2. Compute target score ##
        w_p = R_p.log_map()
        tw_p = torch.cat((t_p, w_p), -1).requires_grad_()
        log_p = self.log_gaussian_on_lie_groups(tw_p, context=[H_in.R, H_in.t, std])
        target_grad = torch.autograd.grad(log_p.sum(), tw_p, only_inputs=True)[0]
        target_score = target_grad.detach()
        #############################

        ## 3. Get diffusion grad ##
        x_in = tw_p.detach().requires_grad_(True)
        H_in = SO3_R3().exp_map(x_in).to_matrix()
        t_in = random_t
        energy = model(H_in, t_in)
        grad_energy = torch.autograd.grad(energy.sum(), x_in, only_inputs=True,
                                          retain_graph=True, create_graph=True)[0]

        ## 4. Compute loss ##
        loss_fn = nn.L1Loss()
        loss = loss_fn(grad_energy, -target_score)/20.

        info = {self.field: energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)

def sigmoid_vfl(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    targets = targets.type_as(inputs)
    focal_weight = targets * (targets > 0.0).float() + \
        alpha * (inputs - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    loss = F.binary_cross_entropy(inputs, targets, reduction="none") * focal_weight # (B, C)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()  # (B, C)
    targets = targets.float()  # (B, C)
    # p = torch.sigmoid(inputs)  # (B, C)
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none") # (B, C)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)  # (B, C)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # # (B, C)
        loss = alpha_t * loss # (B, C)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# def quaternion_step(
#     self,
#     model_output: torch.FloatTensor,
#     timestep: int,
#     sample: torch.FloatTensor,
#     generator=None,
#     return_dict: bool = True,
# ) -> Union[DDPMSchedulerOutput, Tuple]:
#     t = timestep

#     prev_t = self.previous_timestep(t)

#     if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
#         model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
#     else:
#         predicted_variance = None

#     # 1. compute alphas, betas
#     alpha_prod_t = self.alphas_cumprod[t]
#     alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#     beta_prod_t = 1 - alpha_prod_t
#     beta_prod_t_prev = 1 - alpha_prod_t_prev
#     current_alpha_t = alpha_prod_t / alpha_prod_t_prev
#     current_beta_t = 1 - current_alpha_t

#     # 2. compute predicted original sample from predicted noise also called
#     # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
#     if self.config.prediction_type == "epsilon":
#         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
#     elif self.config.prediction_type == "sample":
#         pred_original_sample = model_output
#     elif self.config.prediction_type == "v_prediction":
#         pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
#     else:
#         raise ValueError(
#             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
#             " `v_prediction`  for the DDPMScheduler."
#         )

#     # 3. Clip or threshold "predicted x_0"
#     if self.config.thresholding:
#         pred_original_sample = self._threshold_sample(pred_original_sample)
#     elif self.config.clip_sample:
#         pred_original_sample = pred_original_sample.clamp(
#             -self.config.clip_sample_range, self.config.clip_sample_range
#         )

#     # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
#     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
#     pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
#     current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

#     # 5. Compute predicted previous sample µ_t
#     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
#     pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

#     # 6. Add noise
#     variance = 0
#     if t > 0:
#         prev_t = self.previous_timestep(t)

#         alpha_prod_t = self.alphas_cumprod[t]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#         current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
#         variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
#         variance = torch.clamp(variance, min=1e-20)

#         device = model_output.device
#         variance_noise = randn_tensor(
#             model_output.shape, generator=generator, device=device, dtype=model_output.dtype
#         )

#         variance_noise[...,3:] = nn.functional.normalize(variance_noise[...,3:], dim=-1)
#         variance_noise = variance_noise.reshape(-1, 7)
#         variance_noise[:,None,3:] = variance_noise[:,None,3:] - (torch.bmm(variance_noise[:,None,3:], pred_prev_sample[...,3:].reshape(-1,4)[..., None])) * pred_prev_sample[...,3:].reshape(-1,4)[:, None, :]
#         variance_ = (variance ** 0.5) * variance_noise.reshape(model_output.shape)

#     pred_prev_sample = pred_prev_sample + variance_

#     if not return_dict:
#         return (pred_prev_sample,)

#     return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)



