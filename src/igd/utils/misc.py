import os
import numpy as np
import random
import torch
import skimage.transform
import cv2
import matplotlib.pyplot as plt
def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# def apply_noise(img, noise_type):
#     if noise_type == 'random':
#         noise_type = np.random.choice(['dex','trans','norm','sim_stereo'])
#         print(noise_type)
#     if noise_type == 'dex':
#         return apply_dex_noise(img)
#     elif noise_type =='trans':
#         return apply_translational_noise(img)
#     elif noise_type == 'norm':
#         return apply_gaussian_noise(img)
#     elif noise_type == 'sim_stereo':
#         return add_noise_stereo(img)
#     else:
#         return img

# def apply_noise(img, noise_type):
#     if noise_type == 'random':
#         noise_type = np.random.choice(['dex','trans','norm'])
#         # print(noise_type)
#     if noise_type == 'dex':
#         img = apply_dex_noise(img)
#     elif noise_type =='trans':
#         img = apply_translational_noise(img)
#     elif noise_type == 'norm':
#         img = apply_gaussian_noise(img)
#     # elif noise_type == 'sim_stereo':
#     return add_noise_stereo(img)

def apply_noise(img, noise_type):
    # blur
    # filter_type = np.random.choice(['Gaussian', 'blur'])
    # for _ in range(3):
        # if filter_type == 'Gaussian':
        #     img = cv2.GaussianBlur(img,(5,5),1.5)
        # if filter_type == 'blur':
        # img = cv2.blur(img,(7,7))
    
    # depth noise
    # if noise_type == 'random':
    #     noise_type = np.random.choice(['dex','norm'])
    #     # print(noise_type)
    # if noise_type == 'dex':
        # img = apply_dex_noise(img,gp_rate=1)
    # # elif noise_type =='trans':
    # elif noise_type == 'norm':
    # for _ in range(3):
    #     img = cv2.blur(img, (5,5))
    # img = apply_gaussian_noise(img,sigma=0.005)
    
    # lateral noise
    # img = add_gaussian_shifts(img, std=1)
    img = apply_dex_noise(img)
    # stereo
    # img = add_noise_stereo(img)

    # img = apply_random_hole(img)


    # distortion
    # corruption_type = np.random.choice(['dilate', 'erode', 'none'])
    # if corruption_type == 'dilate':
    #     img = cv2.dilate(img,(7,7))
    # elif corruption_type == 'erode':
    # if np.random.random() < 0.5:
    #     img = cv2.erode(img,(3,7), iterations=3)
    #     img = cv2.dilate(img,(7,3),iterations=3)


    return img


scale_factor  = 100     # converting depth from m to cm 
focal_length  = (386.42556+386.42567)/2  # focal length of the camera used 
baseline_m    = 0.050   # baseline in m 
invalid_disp_ = 99999999.9
dot_pattern_ = cv2.imread("/home/pinhao/Desktop/simkinect/data/kinect-pattern_3x3.png", 0)

def add_noise_stereo(depth_img):
    # if np.random.random() > 0.5:
    #     return depth_img
    #### adding noise #####
    
    depth_interp = add_gaussian_shifts(depth_img, 1/2.)
    # for _ in range(5):
    #     depth_interp = cv2.GaussianBlur(depth_interp, (31,31),1.5)
    disp_= focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0)/8.0

    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

    depth_img = focal_length * baseline_m / out_disp
    # depth_img = cv2.erode(depth_img, kernel=np.ones((5,5)))

    depth_img[out_disp == invalid_disp_] = 0 
    return depth_img

def apply_random_hole(img,
                      gp_scale=4.0,
                      gp_rate=0.5):
    if np.random.rand() < gp_rate:
        return img
    h, w = img.shape[:2]
    gp_sample_height = int(h / gp_scale)
    gp_sample_width = int(w / gp_scale)
    gp_noise = np.random.randn(gp_sample_height, gp_sample_width)+0.5
    gp_noise = cv2.resize(gp_noise, (w,h))
    img[gp_noise<0] = 0.0
    return img
    


def apply_dex_noise(img,
                gamma_shape=1000,
                gamma_scale=0.001,
                gp_sigma=0.005,
                gp_scale=4.0,
                gp_rate=0.5):
    gamma_noise = np.random.gamma(gamma_shape, gamma_scale)
    img = img * gamma_noise
    if np.random.rand() < gp_rate:
        h, w = img.shape[:2]
        gp_sample_height = int(h / gp_scale)
        gp_sample_width = int(w / gp_scale)
        gp_num_pix = gp_sample_height * gp_sample_width
        gp_noise = np.random.randn(gp_sample_height, gp_sample_width) * gp_sigma
        gp_noise = skimage.transform.resize(gp_noise,
                                            img.shape[:2],
                                            order=1,
                                            anti_aliasing=False,
                                            mode="constant")
        
        img += gp_noise
    return img

def apply_translational_noise(img,
                              sigma_p=1,
                              sigma_d=0.005):
    h, w = img.shape[:2]
    hs = np.arange(h)
    ws = np.arange(w)
    ww, hh = np.meshgrid(ws, hs)
    hh = hh + np.random.randn(*hh.shape) * sigma_p
    ww = ww + np.random.randn(*ww.shape) * sigma_p
    hh = np.clip(np.round(hh), 0, h-1).astype(int)
    ww = np.clip(np.round(ww), 0, w-1).astype(int)
    new_img = img[hh, ww]
    # new_img += np.random.randn(*new_img.shape) * sigma_d
    return new_img


def apply_gaussian_noise(img, sigma=0.01):
    img += np.random.randn(*img.shape) * sigma
    return img


def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    corruption_type = np.random.choice(['dilate', 'erode', 'none'])
    gaussian_shifts_ = np.abs(gaussian_shifts.copy())
    gaussian_shifts = cv2.dilate(gaussian_shifts, (31,31),iterations=3)
    gaussian_shifts_[gaussian_shifts<0] *= -1
    gaussian_shifts = gaussian_shifts_
    # gaussian_shifts = cv2.dilate(gaussian_shifts, (11,11),iterations=3)
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp



def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp