# GAN을 활용한 고해상도 탄성파 이미지 생성

## Exmouth 고원 탄성파 이미지 고해상도 복원 작업(흑백)

![My Image](images/gray_scale.png)

## Exmouth 고원 탄성파 이미지 고해상도 복원 작업(컬러)

![My Image](images/seismic_scale.png)

개선된 Reflecotr(반사면)을 확인할 수 있음.

### MODEL

1. Generator
   고해상도 이미지를 만드는 모델
2. Discriminator
   해당 모델이 고해상도로 복원된 이미지인지 원본이미지인지 판단하는 작업을 진행함.

### Compare FaultDetection

#### 원본 이미지

![My Image](images/OG_inf.png)

#### 고해상도 이미지

![My Image](images/HR_inf.png)

저해상도에서 훈련시킨 이미지를 고해상도에서 복원하여 사용하였기 때문에 고해상도 이미지에서 훈련을 시켜볼 필요가 있음.

### TO DO LIST

* [ ] organize the `train.py` code
* [ ] Make and use hyperparameter config.
* [ ] Considering how to handle the checkerboard pattern.
* [ ] Create inference code at the directory level.
* [ ] Compare Fault Detection Score between original seismic vs high-resolution seismic
