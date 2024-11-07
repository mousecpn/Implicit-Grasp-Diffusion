import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from igd.ConvONets.conv_onet.models import decoder
from theseus import SO3
import numpy as np
from igd.utils.transform import Rotation, Transform
import matplotlib.pyplot as plt

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def create_grid_like(t, dim = 0):
    f, h, w, device = *t.shape[-3:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    f, h, w = grid.shape[-3:]
    grid_f, grid_h, grid_w = grid.unbind(dim = dim)

    grid_f = 2.0 * grid_f / max(f - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_f, grid_h, grid_w), dim = out_dim)


class DeformableAttn(nn.Module):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 1,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2
    ):
        super().__init__()
        self.feature_sampler = feature_sampler
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.grid_scale = grid_scale
        self.sp = sample_point_per_axis
        self.embed_dim = feature_dim//num_heads
        self.offset_context = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_k = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_v = nn.Linear(self.feature_dim, self.embed_dim)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = nn.Linear(self.feature_dim, out_dim)
        
        self.scale = self.embed_dim ** -0.5

        self.act = nn.ReLU(inplace=True)
        
        sp = self.sp
        self.to_offset = nn.Linear(self.feature_dim, sp*sp*sp*num_heads*3, bias = True)
        anchor = torch.zeros((sp,sp,sp))
        grid = create_grid_like(anchor)
        grid = anchor + grid
        grid_scaled = normalize_grid(grid, dim=0) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        grid_scaled = grid_scaled.unsqueeze(3).repeat(1,1,1,num_heads,1)
        
        constant_init(self.to_offset, 0.)
        self.to_offset.bias.data = grid_scaled.view(-1)    

        xavier_init(self.to_v, distribution='uniform', bias=0.)
        xavier_init(self.to_k, distribution='uniform', bias=0.)
        xavier_init(self.to_q, distribution='uniform', bias=0.)    
        xavier_init(self.to_out, distribution='uniform', bias=0.) 
        
        self.sample_points = None   

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 3)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        sp = self.sp
        # anchor = torch.zeros((bs*ns,3,sp,sp,sp)).to(query_pos.device)
        # grid = create_grid_like(anchor)
        # grid = anchor + grid
        # grid_scaled = normalize_grid(grid) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        # grid_scaled = grid_scaled.reshape(bs, ns, -1, 3)
        
        # aux_sample_point = query_pos.unsqueeze(2) + grid_scaled # (bs, ns,sp*sp*sp,3)
        # aux_feature = self.feature_sampler(aux_sample_point.reshape(bs*ns, -1, 3), c).reshape(bs, ns,sp*sp*sp, self.num_heads, self.embed_dim)
        
        # feature fusion for offset
        # local_context = self.act(self.offset_context(torch.mean(aux_feature, dim=2,keepdim=True))) + aux_feature
        
        aux_sample_point_offset = self.to_offset(feature).reshape(bs,ns,-1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 1, 3)
        
        self.sample_points = aux_sample_point
        
        # aux_sample_point_offset = aux_sample_point.unsqueeze(3) + offset # (bs, ns, sp*sp*sp, n_heads, 3)
        
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, sp*sp*sp, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
        
        
        k = self.to_k(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        v = self.to_v(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out
    
    
    
class GraspConditionDeformableAttn(nn.Module): ### TO DO num_heads
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 4,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2
    ):
        super().__init__()
        self.feature_sampler = feature_sampler
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.grid_scale = grid_scale
        self.sp = sample_point_per_axis
        self.embed_dim = feature_dim//num_heads
        self.offset_context = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_k = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_v = nn.Linear(self.feature_dim, self.embed_dim)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = nn.Linear(self.feature_dim, out_dim)
        
        self.scale = self.embed_dim ** -0.5

        self.act = nn.ReLU(inplace=True)

        sp = self.sp
        self.to_offset = nn.Linear(self.feature_dim+4, sp*sp*sp*num_heads*3, bias = True)
        anchor = torch.zeros((sp,sp,sp))
        grid = create_grid_like(anchor)
        grid = anchor + grid
        grid_scaled = normalize_grid(grid, dim=0) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        grid_scaled = grid_scaled.unsqueeze(3).repeat(1,1,1,num_heads,1)
        
        constant_init(self.to_offset, 0.)
        self.to_offset.bias.data = grid_scaled.view(-1)    

        xavier_init(self.to_v, distribution='uniform', bias=0.)
        xavier_init(self.to_k, distribution='uniform', bias=0.)
        xavier_init(self.to_q, distribution='uniform', bias=0.)    
        xavier_init(self.to_out, distribution='uniform', bias=0.)    

    def forward(self, grasp, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        query_pos = grasp[..., :3]
        orientation = grasp[..., 3:]
        bs, ns, _ = query_pos.shape
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        feature_ori_condition = torch.cat((feature, orientation), dim=-1) # (bs, ns, feature_dim+4)
        sp = self.sp
        # anchor = torch.zeros((bs*ns,3,sp,sp,sp)).to(query_pos.device)
        # grid = create_grid_like(anchor)
        # grid = anchor + grid
        # grid_scaled = normalize_grid(grid) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        # grid_scaled = grid_scaled.reshape(bs, ns, -1, 3)
        
        # aux_sample_point = query_pos.unsqueeze(2) + grid_scaled # (bs, ns,sp*sp*sp,3)
        # aux_feature = self.feature_sampler(aux_sample_point.reshape(bs*ns, -1, 3), c).reshape(bs, ns,sp*sp*sp, self.num_heads, self.embed_dim)
        
        # feature fusion for offset
        # local_context = self.act(self.offset_context(torch.mean(aux_feature, dim=2,keepdim=True))) + aux_feature
        
        aux_sample_point_offset = self.to_offset(feature_ori_condition).reshape(bs,ns,-1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 1, 3)


        
        # aux_sample_point_offset = aux_sample_point.unsqueeze(3) + offset # (bs, ns, sp*sp*sp, n_heads, 3)
        
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, sp*sp*sp, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
        
        
        k = self.to_k(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        v = self.to_v(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out
    

class GlobalFeatureExtraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GlobalFeatureExtraction, self).__init__()
        self.up_dim_xy = nn.Conv2d(kernel_size=3, in_channels=input_dim, out_channels=hidden_dim, padding=1)
        self.up_dim_xz = nn.Conv2d(kernel_size=3, in_channels=input_dim, out_channels=hidden_dim, padding=1)
        self.up_dim_yz = nn.Conv2d(kernel_size=3, in_channels=input_dim, out_channels=hidden_dim, padding=1)

        self.merge = nn.Conv3d(kernel_size=(5,5,5), in_channels=hidden_dim, out_channels=hidden_dim, dilation=2)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8,dropout=0.1)

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    
    def forward(self, c):
        xy_feat = c['xy']
        yz_feat = c['yz']
        xz_feat = c['xz'] # (bs, c, 40, 40)

        xy_feat = F.relu(self.up_dim_xy(xy_feat))[:, :, :, :, None]
        yz_feat = F.relu(self.up_dim_xy(yz_feat))[:, :, None, :, :]
        xz_feat = F.relu(self.up_dim_xz(xz_feat))[:, :, :, None, :]

        global_feat = xy_feat + yz_feat + xz_feat

        global_feat = self.merge(global_feat)

        
        global_feat = self.pool(global_feat).squeeze()

        return global_feat


class SO3DeformableAttn(nn.Module):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 4,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2
    ):
        super().__init__()
        self.feature_sampler = feature_sampler
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.grid_scale = grid_scale
        self.sp = sample_point_per_axis
        self.embed_dim = feature_dim//num_heads
        self.offset_context = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_k = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_v = nn.Linear(self.feature_dim, self.embed_dim)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = nn.Linear(self.feature_dim, out_dim)
        
        self.scale = self.embed_dim ** -0.5

        self.act = nn.ReLU(inplace=True)

        sp = self.sp
        self.to_offset = nn.Linear(self.embed_dim, 3, bias = False)
        constant_init(self.to_offset, 0.)
        # self.to_offset = nn.Linear(self.feature_dim, sp*sp*sp*num_heads*3, bias = True)
        anchor = torch.zeros((sp,sp,sp))
        grid = create_grid_like(anchor)
        grid = anchor + grid
        grid_scaled = normalize_grid(grid, dim=0) / self.grid_scale # (sp,sp,sp,3)
        grid_scaled = grid_scaled.unsqueeze(3) # (sp,sp,sp, 3)
        self.grid_scaled = nn.Parameter(grid_scaled, requires_grad=True)
        
        # constant_init(self.to_offset, 0.)
        # self.to_offset.bias.data = grid_scaled.view(-1)    

        xavier_init(self.to_v, distribution='uniform', bias=0.)
        xavier_init(self.to_k, distribution='uniform', bias=0.)
        xavier_init(self.to_q, distribution='uniform', bias=0.)    
        xavier_init(self.to_out, distribution='uniform', bias=0.)    

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:]
        query_pos = query_pos[...,:3]
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        sp = self.sp
        
        grid_scaled = self.grid_scaled.unsqueeze(0).repeat(bs*ns,1,1,1,1,1).reshape(bs,ns,-1,3)
        
        # rotation
        rot_SO3 = SO3(quat_scipy2theseus(query_ori.reshape(-1,4))).to_matrix().reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        grid_scaled = torch.einsum('bnpd,bngd->bngp', rot_SO3, grid_scaled)
        
        anchor_sample_point = query_pos.unsqueeze(2) + grid_scaled # (bs, ns,sp*sp*sp,3)
        
        anchor_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs, ns,sp*sp*sp, self.num_heads, self.embed_dim)
        
        # feature fusion for offset
        context_anchor_feature = self.act(self.offset_context(torch.mean(anchor_feature, dim=2, keepdim=True))) + anchor_feature # (bs, ns, sp*sp*sp, self.num_heads, self.embed_dim)
        
        anchor_offset = self.to_offset(context_anchor_feature).reshape(bs, ns, -1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)

        sample_point = anchor_offset+anchor_sample_point.reshape(bs, ns, -1, 1, 3)
                
        sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, sp*sp*sp, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
        
        
        k = self.to_k(sample_feature).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        v = self.to_v(sample_feature).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out

class GraspSO3DeformableAttn(SO3DeformableAttn):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 4,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2,
        zero_offset = False,
        fixed_control_points = False
    ):
        super().__init__(
            feature_dim = feature_dim,
            out_dim= out_dim,
            feature_sampler=feature_sampler,
            num_heads = num_heads,
            dropout = dropout,
            grid_scale = grid_scale,
            sample_point_per_axis = sample_point_per_axis
        )
        
        control_points = [[0.0, 0.0, 0.0], # wrist
                          [0.0, -0.04, 0.05], # right_finger_tip
                          [0.0, 0.04, 0.05],  # left_finger_tip
                          [0.0, -0.04, 0.00], # palm
                          [0.0, 0.04, 0.00],
                          ]
        
        ### complicate gripper #######
        # num_inter_points = 5
        # palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        # palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
        # finger_z = np.linspace(0.0, 0.05, num_inter_points)
        # left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
         ### complicate gripper #######
        
        ### intersection gripper #######
        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_z = np.linspace(0.0, 0.05, num_inter_points)
        palm_x = np.zeros(1)
        control_points = np.stack(np.meshgrid(palm_x, palm_y,palm_z),axis=-1).reshape(-1,3)
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
        ### intersection gripper #######

        
        control_points = torch.from_numpy(np.array(control_points).astype('float32')) / 0.3
        if fixed_control_points:
            self.register_buffer('control_points', control_points)
        else:
            self.control_points = nn.Parameter(control_points)
        self.zero_offset = zero_offset
        self.to_offset = nn.Linear(self.feature_dim, self.num_heads*3, bias = False)
        constant_init(self.to_offset, 0.)
        # 
        
    def forward(self, query_pos, c, voxel_grid=None):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:]
        query_pos = query_pos[...,:3]
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        
        # sp = self.sp
        # anchor = torch.zeros((bs*ns,3,sp,sp,sp)).to(query_pos.device)
        # grid = create_grid_like(anchor)
        # grid = anchor + grid
        # grid_scaled = normalize_grid(grid) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        # grid_scaled = grid_scaled.reshape(bs, ns, -1, 3) # grasp2offset
        
        # rotation
        rot_SO3 = SO3(quat_scipy2theseus(query_ori.reshape(-1,4))).to_matrix().reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        # control_points_ = control_points.clone()
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        # control_points = torch.einsum('bndp,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

        # print_grasp(voxel_grid[0].detach().cpu().numpy(), rot_SO3[0,0].detach().cpu().numpy(), (query_pos[0,0].detach().cpu().numpy()+0.5)*0.3)

        
        
        #### test ########
        # R_z_ = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        # R_z = torch.from_numpy(R_z_.as_matrix().astype('float32')).to(rot_SO3.device)
        # control_points_z = torch.einsum('pd, bngd -> bngp', R_z , control_points_)
        # control_points_2 = torch.einsum('bnpd, bngd -> bngp', rot_SO3 , control_points_z)
        
        # rot_SO3_2 = torch.einsum('bngd, dp -> bngp', rot_SO3 , R_z)
        # # rot_SO3_2 = torch.einsum('bngd, dp -> bngp', rot_SO3_2 , R_z)
        # control_points_4 = torch.einsum('bnpd, bngd -> bngp', rot_SO3_2 , control_points_z)
        
        # term = SO3()
        # term.update(rot_SO3_2.reshape(-1,3,3))
        # term = term.to_quaternion().reshape(bs,ns,4)
        
        # rot_SO3_z_ = Rotation.from_matrix(rot_SO3.reshape(-1,3,3).detach().cpu().numpy())
        # R_z_ =  Rotation.from_matrix(R_z_.as_matrix().repeat(bs*ns).reshape(-1,3,3))
        # rot_SO3_z_ = (rot_SO3_z_ * R_z_).as_quat()
        # rot_SO3_z = torch.from_numpy(rot_SO3_z_.astype('float32')).to(rot_SO3.device)
        # rot_SO3_z = SO3(quat_scipy2theseus(rot_SO3_z.reshape(-1,4))).to_matrix().reshape(bs,ns,3,3)

        # control_points_3 = torch.einsum('bnpd, bngd -> bngp', rot_SO3_z.reshape(bs,ns,3,3) , control_points_)
        # print()
        #### test ########
                
        # feature fusion for offset
        if self.zero_offset:
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            """
            anchor_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs, ns, -1, self.num_heads, self.embed_dim)
            context_anchor_feature = self.act(self.offset_context(torch.mean(anchor_feature, dim=2, keepdim=True))) + anchor_feature # (bs, ns, sp*sp*sp, self.num_heads, self.embed_dim)
            anchor_offset = self.to_offset(context_anchor_feature).reshape(bs, ns, -1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)
            sample_point = anchor_offset+anchor_sample_point.reshape(bs, ns, -1, 1, 3)
            sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, -1, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
            """
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
                    
            sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
            
        
        k = self.to_k(sample_feature).transpose(1,2) # (bs*ns, n_heads, ncp, embed_dim)
        v = self.to_v(sample_feature).transpose(1,2) # (bs*ns, n_heads, ncp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out

def quat_theseus2scipy(quat):
    """
    from (w,x,y,z) ==> (x,y,z,w)
    """
    w = quat[...,:1]
    xyz = quat[...,1:]
    quat = torch.cat((xyz,w), dim=-1)
    return quat 


def print_grasp(voxel_grid, ori, pos):
    ax = plt.figure().add_subplot(projection='3d')
    points = np.stack(np.where((voxel_grid.squeeze()>0.0) & (voxel_grid.squeeze()<0.1)),axis=-1)
    # ax.scatter(points[:,0], points[:,1], points[:,2], c='green')

    num_inter_points = 5
    # palm_y = np.linspace(-0.04, 0.04, num_inter_points)
    # palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
    # finger_z = np.linspace(0.0, 0.05, num_inter_points)
    # left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
    # right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
    # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)

    palm_y = np.linspace(-0.04, 0.04, num_inter_points)
    palm_z = np.linspace(0.0, 0.05, num_inter_points)
    palm_x = np.zeros(1)
    control_points = np.stack(np.meshgrid(palm_x, palm_y,palm_z),axis=-1).reshape(-1,3)

    # R = ori.as_matrix() # (3,3)
    R = ori # (3,3)

    control_points2 = (R @ control_points.transpose(1,0)).transpose(1,0) / 0.3 * 40

    control_points2 += (pos.reshape(-1,3) / 0.3) * 40

    ax.scatter(points[:,0], points[:,1], points[:,2], c='green')
    ax.scatter(control_points2[:,0], control_points2[:,1], control_points2[:,2], c='red', s=30)

    plt.show()

def quat_scipy2theseus(quat):
    """
    from (x,y,z,w) ==> (w,x,y,z) 
    """
    w = quat[...,-1:]
    xyz = quat[...,:3]
    quat = torch.cat((w,xyz), dim=-1)
    return quat
    
if __name__=="__main__":
    context = {}
    bs = 4
    c_dim = 32
    x_dim=y_dim=z_dim = 40 # h-->x, w-->y, z-->z
    xy_feat = xz_feat = yz_feat = torch.randn((bs, c_dim, x_dim, y_dim))
    context['xy'] = xy_feat
    context['xz'] = xz_feat
    context['yz'] = yz_feat

    pos = torch.rand((bs,2,3))*2 - 1
    grasp = torch.randn((bs,2,7))
    grasp[...,3:] = nn.functional.normalize(grasp[...,3:], dim=-1)


    global_extractor = GlobalFeatureExtraction(32, 128)
    feat = global_extractor(context)

    attn = DeformableAttn(96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=1)
    feat = attn(pos, context)
    
    # graspattn = GraspConditionDeformableAttn(96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=1)
    # graspattn = GraspSO3DeformableAttn(96, out_dim=96, feature_sampler=decoder.FCDecoder(), num_heads=1)
    # feat = graspattn(grasp, context)

    # t = torch.zeros((128,3,2,2,2))
    # grid = create_grid_like(t)
    # grid = t + grid
    # norm_grid = normalize_grid(grid)
    print()


        
        
        
        