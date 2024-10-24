import torch
import torch.nn.functional as F
from datetime import datetime
import logging
from prettytable import PrettyTable
import numpy as np
from skimage.util.shape import view_as_windows
import scipy.io as sio
import scipy.signal as signal
from itertools import product

def _downscale(images, K):
    """Differentiable image downscaling by a factor of K for 1-channel images"""
    # KxK의 평균 필터 생성 (1채널)
    arr = torch.zeros((K, K, 1, 1), dtype=torch.float32, device=images.device)  # 이미지와 동일한 디바이스로 생성
    arr[:, :, 0, 0] = 1.0 / (K * K)  # 평균값 설정
    downscale_weight = arr.permute(2, 3, 0, 1)  # Conv2d는 (out_channels, in_channels, height, width) 형식 필요

    # Conv2d로 다운스케일링 수행
    downscaled = F.conv2d(images, downscale_weight, stride=K, padding=K // 2 - 1)
    return downscaled

def create_generator_loss(disc_output, gene_output, features, gene_l1_factor):
    # Discriminator를 속였는가?
    cross_entropy = F.binary_cross_entropy_with_logits(disc_output, torch.ones_like(disc_output))
    gene_ce_loss = cross_entropy.mean()

    # 결과가 특징과 얼마나 비슷한가?
    K = gene_output.shape[2] // features.shape[2]  # height 기준으로 비율 계산
    assert K in (2, 4, 8), "K must be 2, 4, or 8"  # K 값 확인
    downscaled = _downscale(gene_output, K)
    
    gene_l1_loss = F.l1_loss(downscaled, features)

    # 최종 손실 계산
    gene_loss = (1.0 - gene_l1_factor) * gene_ce_loss + gene_l1_factor * gene_l1_loss
    return gene_loss

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # 진짜 입력을 올바르게 인식했는가?
    cross_entropy_real = F.binary_cross_entropy_with_logits(disc_real_output, torch.ones_like(disc_real_output))
    disc_real_loss = cross_entropy_real.mean()

    # 가짜 입력을 올바르게 인식했는가?
    cross_entropy_fake = F.binary_cross_entropy_with_logits(disc_fake_output, torch.zeros_like(disc_fake_output))
    disc_fake_loss = cross_entropy_fake.mean()

    return disc_real_loss, disc_fake_loss

import os

def setup_logging(log_dir = './'):
    log_dir = log_dir  # 로그를 저장할 폴더 경로
    os.makedirs(log_dir, exist_ok=True)  # logs 폴더가 없으면 생성
    log_filename = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Log setup complete.')

# 하이퍼파라미터 로깅 함수
def log_hyperparameters(params):
    table = PrettyTable()
    table.field_names = ["Hyperparameter", "Value"]
    
    for key, value in params.items():
        table.add_row([key, value])
    
    logging.info("Hyperparameters:\n" + str(table))
    print("Hyperparameters:\n" + str(table))

def normalize_seis(seismic):
    min = seismic.min()
    max = seismic.max()
    return (seismic - min) / (max - min)


def get_sliding_wnd_params(img_sz, patch_sz, step):
    # For example, if patch_sz = 257, step = 128 ...
    # and img_sz = (3174, 1537) ...
    # (Image size is .T of ndarray shape because x and y are col, row each)
    overlap_sz = patch_sz - step # overlap_sz = 129
    img_width, img_height = img_sz # img_width = 3174, img_height = 1537

    horizontal_cnt = int(np.ceil((img_width - overlap_sz) / step)) # No. of horizontal step to cover whole image
    new_img_width = step * horizontal_cnt + overlap_sz # New width w/ padding
    pad_l = int((new_img_width - img_width) / 2) # left padding
    pad_r = new_img_width - img_width - pad_l # right padding

    vertical_cnt = int(np.ceil((img_height - overlap_sz) / step)) 
    new_img_height = step * vertical_cnt + overlap_sz
    pad_t = int((new_img_height - img_height) / 2) # top padding
    pad_b = new_img_height - img_height - pad_t # bottom padding

    return ((pad_t, pad_b), (pad_l, pad_r)), (horizontal_cnt, vertical_cnt)


def get_sliding_wnd_patches(img, padding, patch_sz, step):
    padded = np.pad(img, padding, 'reflect')
    patches = view_as_windows(padded, (patch_sz, patch_sz), step=step)
    patches = patches.reshape((-1, patch_sz, patch_sz))
    return patches

def spline_window(wnd_sz, power=2):
    '''
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    '''
    intersection = int(wnd_sz/4)
    wnd_outer = (abs(2*(signal.triang(wnd_sz))) ** power)/2
    wnd_outer[intersection:-intersection] = 0

    wnd_inner = 1 - (abs(2*(signal.triang(wnd_sz) - 1)) ** power)/2
    wnd_inner[:intersection] = 0
    wnd_inner[-intersection:] = 0

    wnd = wnd_inner + wnd_outer
    wnd = wnd / np.average(wnd)
    return wnd

cached_2d_windows = dict()

def window_2d(wnd_sz, power=2):
    '''
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    '''
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(wnd_sz, power)
    if key in cached_2d_windows:
        wnd = cached_2d_windows[key]
    else:
        wnd = spline_window(wnd_sz, power)
        wnd = np.expand_dims(np.expand_dims(wnd, -1), -1)
        wnd = wnd * wnd.transpose(1, 0, 2)
        cached_2d_windows[key] = wnd
    return wnd
    
def recover_img_from_patches(patches, img_sz, padding, overlap_sz):
    assert len(patches.shape) == 4
    # patches : vert. cnt, hor. cnt, patch_sz, patch_sz
    weights_each = window_2d(wnd_sz=patches.shape[3], power=2).squeeze()
    img_height, img_width = img_sz
    img = np.zeros(img_sz, dtype=patches.dtype)
    divisor = np.zeros(img_sz, dtype=patches.dtype)

    img = np.pad(img, padding, 'reflect')
    divisor = np.pad(divisor, padding, 'reflect')

    num_height, num_width, patch_height, patch_width = patches.shape

    step_width = patch_width - overlap_sz
    step_height = patch_height - overlap_sz

    for n_y, n_x in product(range(num_height), range(num_width)):
        patch = patches[n_y, n_x]
        x_i, y_i = n_x * step_width, n_y * step_height
        x_f, y_f = n_x * step_width + patch_width, n_y * step_height + patch_height
        img[y_i:y_f, x_i:x_f] += patch
        divisor[y_i:y_f, x_i:x_f] += weights_each
        # divisor[y_i:y_f, x_i:x_f] += 1 -- old code

    recovered = img / divisor
    pad_t = padding[0][0]
    pad_l = padding[1][0]
    return recovered[pad_t:pad_t + img_height, pad_l:pad_l + img_width]
