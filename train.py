import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import random
from models.Generator import GeneratorModel
from models.Discriminator import DiscriminatorModel
import os
from os import listdir
from os.path import splitext
import numpy as np
import tqdm
from utils import *
from utils import _downscale

#########################################################로깅 세팅###########################################################
setup_logging()
###################################################하이퍼파라미터 세팅 & 기록 ################################################
hyperparameters = {
        "learning_rate_g": 0.001,
        "learning_rate_d": 0.001,
        "batch_size": 64,
        "num_epochs": 10,
        "optimizer_g": "Adam",
        "optimizer_d": "Adam",
        "loss_function": "BCEWithLogitsLoss"
    }
log_hyperparameters(hyperparameters)
###########################################################################################################################

#####################################################데이터셋 정의##########################################################
class SeismicDataset(Dataset):
    def __init__(self, imgs_dir, downscale_factor=4):
        self.images_dir = imgs_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.downscale_factor = downscale_factor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img = np.load("{}/{}.npy".format(self.images_dir, idx))

        # MIN-MAX SCALING
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min != 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img / 255.0  # IF MIN == MAX - DIVIDE BY 255

        img = np.expand_dims(img, axis=0)  # 채널 차원 추가

        # Torch Tensor로 변환
        img = torch.tensor(img, dtype=torch.float32)

        # 원본 이미지를 레이블로 설정
        label = img.clone()

        # 입력 이미지로 사용할 저해상도 이미지 생성
        low_res_img = F.interpolate(img.unsqueeze(0), scale_factor=1/self.downscale_factor, mode='bilinear', align_corners=False)
        low_res_img = low_res_img.squeeze(0)

        # print(low_res_img.shape)
        # print(label.shape) # 위의 저해상도 필터가 정상적으로 작동했다면 label의 크기가 low_res_img의 4배의 크기를 가지고 있을 것
        
        # 저해상도 이미지를 원본 크기로 복원
        low_res_img_restored = F.interpolate(low_res_img.unsqueeze(0), scale_factor=self.downscale_factor, mode='bilinear', \
                                             align_corners=False)
        low_res_img_restored = low_res_img_restored.squeeze(0)

        return low_res_img_restored, label  # 복원된 저해상도 이미지와 원본 이미지(레이블) 반환
##################################################################################################################################



##################################################### DataLoader 사용 #############################################################
def create_data_loader(imgs_dir, batch_size=32, shuffle=True, subset_size=None):
    dataset = SeismicDataset(imgs_dir)

    # 서브셋 크기를 지정한 경우
    if subset_size is not None:
        indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
        dataset = Subset(dataset, indices)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
#################################################################################################################################

############################################## GPU 확인 및 모델 초기화 ###########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##################################################### 경로 설정 #################################################################
imgs_dir = '/data2/seismic_HYUL/dataset/thebe_processed_128_64/train/seismic'  # 이미지 경로를 설정
data_loader = create_data_loader(imgs_dir, batch_size=8, subset_size = 10000)
save_dir = "/data2/High_resolution/Result"

######################## 모델 초기화(input channel과 output channel은 흑백이면 1-channel 컬러면 3-channel)##########################
generator = GeneratorModel(input_channels=1, output_channels=1).to(device)
discriminator = DiscriminatorModel(input_channels=1).to(device)

######################################################## 옵티마이저 선정###########################################################
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
##################################################################################################################################

############################################### 저해상도 이미지에 약간의 노이즈 추가################################################
def add_noise_to_features(train_features, noise_level):
    """
    train_features : noise를 추가할 이미지
    noise_level : noise 추가 비율
    """
    # 주어진 train_features에 노이즈 추가
    noise = torch.randn_like(train_features) * noise_level
    noisy_train_features = train_features + noise
    return noisy_train_features
####################################################################################################################################


##################################################### 모델 저장 함수 #################################################################
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
####################################################################################################################################



########################################################훈련 시작!###################################################################
num_epochs = 10 
for epoch in range(num_epochs):
    for i, (low_res_image, real_images) in enumerate(tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        real_images = real_images.to(device)  # 레이블(원본 이미지)을 GPU로 이동
        low_res_image = low_res_image.to(device)  # 입력 이미지를 GPU로 이동

        # 진짜 데이터 레이블
        real_labels = torch.ones(real_images.size(0), 1, device=device)  # 진짜 레이블 (1)
        fake_labels = torch.zeros(real_images.size(0), 1, device=device)  # 가짜 레이블 (0)

        # Discriminator 훈련
        optimizer_D.zero_grad()
        
        # 진짜 이미지에 대한 Discriminator 출력
        outputs_real = discriminator(real_images)
        d_loss_real = F.binary_cross_entropy_with_logits(outputs_real, real_labels)

        # 가짜 이미지 생성 (Generator가 생성)
        fake_images = generator(low_res_image)
        fake_images_down_scale = _downscale(fake_images, 4)
        outputs_fake = discriminator(fake_images_down_scale.detach())  # detach()로 기울기 계산 방지
        d_loss_fake = F.binary_cross_entropy_with_logits(outputs_fake, fake_labels)

        # 총 Discriminator 손실 및 가중치 업데이트
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Generator 훈련
        optimizer_G.zero_grad()

        # 가짜 이미지에 대해 Discriminator의 출력값 사용 (기울기 계산 필요)
        outputs_fake_for_G = discriminator(fake_images)
        g_loss = create_generator_loss(outputs_fake_for_G, fake_images, real_images, gene_l1_factor=0.9)  # Generator 손실 계산
        
        g_loss.backward()
        optimizer_G.step()

    logging.info(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    # 모델 저장
    model_save_path = f"{save_dir}/generator_epoch_{epoch+1}.pth"
    save_model(generator, model_save_path)
#####################################################################################################################################