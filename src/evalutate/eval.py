import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from dataset import Dataset
from net_load import load
from transform import ToTensor, Normalization

## 환경 세팅
result_dir = './result'
ckpt_dir = './checkpoint'
data_dir = './image' # 찍거나 선택한 이미지를 업로드할 위치
return_dir = './return'
batch_size = 1
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 부가적인 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)
#############################################################################

##1. eval 진행
print('start evaluation!')

# 이미지 경로와 체크포인트 경로 설정
image_path = './image/input.jpg'    # 이미지 경로
ckpt_path = './checkpoint/model_epoch60.pth' # 모델 체크포인트 경로

# 이미지 불러오기
image = Image.open(image_path).convert('L') # 흑백 이미지로 변환

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
image = transform(image)
image = image.unsqueeze(0) # 배치 차원 추가

# 모델 불러오기
net = UNet()
net = load(ckpt_dir,net)

with torch.no_grad():
    net.eval()

    output = net(image)
    
    # 테스트 결과 저장하기
    input = fn_tonumpy(fn_denorm(image, mean=0.5, std=0.5))
    output = fn_tonumpy(fn_class(output))
    
    for j in range(input.shape[0]):
      id = j
      
      plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
      plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

      np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
      np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
            
lst_data = os.listdir(os.path.join(result_dir, 'numpy'))

#lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

#lst_label.sort()
lst_input.sort()
lst_output.sort()

# #############################################################################

##2. 결과를 이미지화
for id in range(1):
  #label = np.load(os.path.join(result_dir,"numpy", lst_label[id]))
  input = np.load(os.path.join(result_dir,"numpy", lst_input[id]))
  output = np.load(os.path.join(result_dir,"numpy", lst_output[id]))

  # 원본 이미지와 정답지를 결합
  combined_image = np.zeros_like(input)
  combined_image[:, :] = input          # 원본 이미지 복사
  combined_image[output == 1 ] = 0       # 정답지를 흰색으로 설정 (255는 흰색을 의미)

  ## 플롯 그리기
  plt.subplot(131)
  plt.imshow(input, cmap='gray')
  plt.title('input')

  plt.subplot(132)
  plt.imshow(output, cmap='gray')
  plt.title('output')

  plt.subplot(133)
  plt.imshow(combined_image, cmap='gray')  # 그레이스케일 이미지로 표시
  plt.title('result')

  plt.imsave(os.path.join(return_dir, 'output.png'), combined_image, cmap='gray')
  #plt.show()