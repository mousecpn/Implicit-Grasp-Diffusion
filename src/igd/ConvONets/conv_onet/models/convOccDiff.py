import torch
import torch.nn as nn
from torch import distributions as dist
from igd.ConvONets.conv_onet.models import decoder
from igd.ConvONets.conv_onet.models.diffusion import SO3_R3
from igd.ConvONets.conv_onet.models.diffusion_loss import Diffusion, FocalLoss, sigmoid_focal_loss
import torch.nn.functional as F
import numpy as np
from igd.ConvONets.conv_onet.models.deformable_sampling_attention import DeformableAttn, GraspSO3DeformableAttn
from igd.utils.transform import Rotation, Transform

class ConvolutionalOccupancyDiffuser(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()
        # self.feature_sampler = decoder.FCDecoder()   
        sp = 2
        self.quality = False
        grid_scale = 80.
        num_heads = 1
        self.decoder_qual = decoder.LocalDecoder(dim=3, out_dim=1, c_dim=32, hidden_size=32, concat_feat=True, feature_sampler=DeformableAttn(96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=num_heads, grid_scale=grid_scale, sample_point_per_axis=sp))
        self.decoder_grasp_qual = decoder.LocalDecoder(dim=3, out_dim=1, c_dim=32, hidden_size=32, concat_feat=True, feature_sampler=GraspSO3DeformableAttn(96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=num_heads, zero_offset=True, fixed_control_points=False))
        self.feature_sampler = DeformableAttn(feature_dim=96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=num_heads, grid_scale=grid_scale, sample_point_per_axis=sp)
        self.decoder_rot = decoder.TimeLocalDecoder(dim=7, out_dim=7, c_dim=96, hidden_size=128, concat_feat=True, quality=self.quality)
        self.decoder_width = decoder.LocalDecoder(dim=3, out_dim=1, c_dim=32, hidden_size=32, concat_feat=True)
        self.grasp_sampler = Diffusion(schedulers="DDPM", condition_mask=[1,1,1,0,0,0,0],  prediction_type='epsilon',beta_schedule='linear', quality=self.quality)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf
        self.focal_loss = FocalLoss(2, 0.25)

    def forward(self, inputs, p, target=None, p_tsdf=None, eval_on_dataset=False, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            target: label, rotations, width, occ_value
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        c = self.encode_inputs(inputs)
        self.outputs = {}
        self.voxel_grids = inputs

        if target is not None:
            label, rot_gt, width_gt, occ_value = target

            loss_dict = {}
        if self.training is False:
            noise = torch.randn((self.batch_size, self.sample_num, 6), device=inputs.device)
                

        if self.training:
            loss_qual, loss_rot, loss_width = self.decode_train(p, c, target)
            loss_dict['loss_qual'] = loss_qual.mean()
            # loss_dict['loss_pos'] = loss_pos.mean()
            loss_dict['loss_rot'] = loss_rot.mean()
            loss_dict['loss_width'] = (loss_width[label.bool()]).mean()
        else:
            qual, rot, width = self.decode(p, c, target=target)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            if self.training:
                loss_occ = self._occ_loss_fn(torch.sigmoid(tsdf), occ_value)
                loss_dict['loss_occ'] = loss_occ.mean()
                return loss_dict
            return qual, rot, width, tsdf
        else:
            if self.training:
                return loss_dict
            return qual, rot, width
    

    def encode_inputs(self, inputs):
        ''' Encodes the input.optimizer

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c


    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode_train(self, p, c, target, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        label, rot_gt, width_gt, occ_value = target

        feature = self.feature_sampler(p, c)

        positive_grasp = torch.cat((p.repeat(1,2,1), rot_gt), dim=-1)
        antipodal_positive_grasp = positive_grasp.clone()
        antipodal_positive_grasp[...,3:] = -antipodal_positive_grasp[...,3:]
        all_grasp = torch.cat((positive_grasp, antipodal_positive_grasp),dim=1)

        model_input = {}
        model_input['x_ene_pos'] = all_grasp
        model_input['visual_context'] = feature.repeat(1,4,1)

        if self.quality:
            loss_rot, reg_qual, reg_label = self.grasp_sampler.loss_fn(self.decoder_rot, model_input)
            reg_qual = torch.sigmoid(reg_qual)
        else:
            loss_rot = self.grasp_sampler.loss_fn(self.decoder_rot, model_input)
        
        width = self.decoder_width(all_grasp, c, **kwargs)
        width_gt = width_gt.unsqueeze(1).repeat(1,all_grasp.shape[1])
        loss_width = self._width_loss_fn(width, width_gt).mean(-1)
        
        loss_rot = loss_rot[label.bool()]
        label = label.unsqueeze(1).repeat(1,all_grasp.shape[1])

        
        qual = self.decoder_qual(all_grasp, c, **kwargs)
        qual = torch.sigmoid(qual)
        loss_qual = self._qual_loss_fn(qual.reshape(-1), label.reshape(-1)).mean()
        
        neg_sample, neg_label = self.massive_negetive_sampling(all_grasp, label)

        all_grasp = torch.cat((all_grasp, neg_sample), dim=1)
        label = torch.cat((label, neg_label), dim=1)

        grasp_qual = self.decoder_grasp_qual(all_grasp, c, **kwargs)
        grasp_qual = torch.sigmoid(grasp_qual)

        loss_qual += sigmoid_focal_loss(grasp_qual.reshape(-1), label.reshape(-1), reduction='mean')

        self.outputs['qual'] = (grasp_qual[:,0]*qual[:,0]).sqrt()

        return loss_qual, loss_rot, loss_width


    def massive_negetive_sampling(self, grasp, label):
        neg_samples = torch.zeros_like(grasp[:,:0,:])
        # neg_sample = grasp.clone()
        bs, ns = grasp.shape[0], grasp.shape[1]
        sample_type = np.random.choice([0,1])
        trans_perturb_level = 0.1
        rot_perturb_level = 0.5
        num_trans_samples = 10
        num_rotations = 6
        # neg_label = label.clone()

        yaws = np.linspace(0.0, np.pi, num_rotations)
        for yaw in yaws[1:-1]:
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_quat()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = neg_rot.reshape(bs,ns,4)
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

        for i in range(num_trans_samples):
            neg_sample = grasp.clone()
            noise = torch.randn_like(grasp[...,:3]) * trans_perturb_level
            neg_sample[..., :3] += noise
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)
            yaws = np.linspace(0.0, np.pi, num_rotations)
            yaw = np.random.choice(yaws[1:-1])
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_quat()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = neg_rot.reshape(bs,ns,4)
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

        return neg_samples, torch.zeros_like(neg_samples[...,0])

    def decode(self, p, c, target=None, sample_rounds = 1, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p = p.reshape(self.batch_size,self.sample_num,-1)

        feature = self.feature_sampler(p, c)

        if target is not None:
            label, rot_gt, width_gt, occ_value = target
            idx = torch.randint(0,2, (1,))[0]
            rot_gt_ = rot_gt[:,idx,:]
            data = torch.cat((p, rot_gt_.reshape(self.batch_size, self.sample_num, -1)),dim=-1)
            if self.quality:
                grasp, reg_qual = self.grasp_sampler.sample_data(p, feature, self.decoder_rot)
                reg_qual = torch.sigmoid(reg_qual)
            else:
                grasp = self.grasp_sampler.sample_data(p, feature, self.decoder_rot)
            width = self.decoder_width(data, c, **kwargs)
            grasp_qual = self.decoder_grasp_qual(data, c, **kwargs).reshape(self.batch_size, self.sample_num)
            grasp_qual = torch.sigmoid(grasp_qual)
            qual = self.decoder_qual(p, c, **kwargs)
            qual = torch.sigmoid(qual)
            qual = (qual*grasp_qual).sqrt()
            rot = grasp[...,3:]

        else:
            qual, rot, width = self.decode_inference(p, c, sample_rounds = sample_rounds, low_th=0.002)

        return qual, rot, width
    
    def decode_inference(self, p, c, sample_rounds = 1, low_th=0.1, **kwargs):
        assert self.batch_size==1, "batch size should be 1 in this mode" 
        p = p.reshape(self.batch_size, self.sample_num,-1)
        
        feature = self.feature_sampler(p, c)
        feature_dim = feature.shape[-1]
        
        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        
        mask = (qual > low_th)
        
        p_postive = p[mask].reshape(self.batch_size, -1, 3)
        
        
        # loop mode
        for i in range(sample_rounds):
            grasp = self.grasp_sampler.sample_data(p_postive, feature[mask].reshape(self.batch_size, -1, 96), self.decoder_rot)
            grasp[...,3:] = nn.functional.normalize(grasp[...,3:], dim=-1)
            grasp_qual = self.decoder_grasp_qual(grasp, c, **kwargs)
            if i == 0:
                last_grasp = grasp
                last_grasp_qual = grasp_qual
                continue
            comparing_grasp_qual = torch.stack((last_grasp_qual, grasp_qual), dim=1) # (bs, 2, pos_ns)
            comparing_grasp = torch.stack((last_grasp, grasp), dim=1) # (bs, 2, pos_ns, 7)
            last_grasp_qual, indices = torch.max(comparing_grasp_qual, dim=1)
            indices = indices.reshape(self.batch_size, 1, -1, 1).repeat(1,1,1,7)
            last_grasp = comparing_grasp.reshape(self.batch_size, 2, -1, 7).gather(1, indices)
            last_grasp = last_grasp.squeeze(1)
    
        grasp = torch.randn(self.batch_size, self.sample_num, 7).to(last_grasp.device)
        grasp[...,3:] = nn.functional.normalize(grasp[...,3:], dim=-1)
        grasp[mask] = last_grasp.reshape(-1,7)
        grasp_qual = torch.zeros_like(qual)
        grasp_qual[mask] = torch.sigmoid(last_grasp_qual.reshape(-1))

        rot = grasp[...,3:]
        
        width = self.decoder_width(grasp, c, **kwargs)
        
        # qual = grasp_qual
        qual = (qual*grasp_qual).sqrt()
        # qual = reg_qual
        
        return qual, rot, width

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def _qual_loss_fn(self, pred, target):
        # return sigmoid_vfl(pred, target, reduction='none')
        # return sigmoid_focal_loss(pred, target, reduction='none')
        return F.binary_cross_entropy(pred, target, reduction="none")


    def _quat_loss_fn(self, pred, target):
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


    def _width_loss_fn(self, pred, target):
        return F.mse_loss(40 * pred, 40 * target, reduction="none")

    def _occ_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)