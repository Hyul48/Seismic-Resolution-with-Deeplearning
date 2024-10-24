import torch
import torch.nn.functional as F
from datetime import datetime
import logging
from prettytable import PrettyTable

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
    log_dir = "./logs"  # 로그를 저장할 폴더 경로
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
