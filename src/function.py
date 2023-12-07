import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from evalutate.dataset import Dataset
from evalutate.model import UNet
from evalutate.net_load import load
from evalutate.transform import Normalization, ToTensor

import matplotlib



def evaluate():
    
    matplotlib.use('agg')
    
    ## 환경 세팅
    result_dir = './evalutate/result'
    ckpt_dir = './evalutate/checkpoint'
    data_dir = './evalutate/image'  # 찍거나 선택한 이미지를 업로드할 위치
    return_dir = './evalutate/return'
    batch_size = 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 부가적인 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)
    #############################################################################

    ##1. eval 진행
    print('start evaluation!')

    # 이미지 경로와 체크포인트 경로 설정
    image_path = './evalutate/image/input.png'  # 이미지 경로
    ckpt_path = './evalutate/checkpoint/model_epoch60.pth'  # 모델 체크포인트 경로

    # 이미지 불러오기
    image = Image.open(image_path).convert('L')  # 흑백 이미지로 변환

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가

    # 모델 불러오기
    net = UNet()
    net = load(ckpt_dir, net)

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

    # lst_label = [f for f in lst_data if f.startswith('label')]
    lst_input = [f for f in lst_data if f.startswith('input')]
    lst_output = [f for f in lst_data if f.startswith('output')]

    # lst_label.sort()
    lst_input.sort()
    lst_output.sort()

    # #############################################################################

    ##2. 결과를 이미지화
    for id in range(1):
        # label = np.load(os.path.join(result_dir,"numpy", lst_label[id]))
        input = np.load(os.path.join(result_dir, "numpy", lst_input[id]))
        output = np.load(os.path.join(result_dir, "numpy", lst_output[id]))

        # 원본 이미지와 정답지를 결합
        combined_image = np.zeros_like(input)
        combined_image[:, :] = input  # 원본 이미지 복사
        combined_image[output == 1] = 0  # 정답지를 흰색으로 설정 (255는 흰색을 의미)

        ## 플롯 그리기
        plt.subplot(131)
        # plt.imshow(input, cmap='gray')
        plt.title('input')

        plt.subplot(132)
        # plt.imshow(output, cmap='gray')
        plt.title('output')

        plt.subplot(133)
        # plt.imshow(combined_image, cmap='gray')  # 그레이스케일 이미지로 표시
        plt.title('result')

        plt.imsave(os.path.join(return_dir, 'output.png'), combined_image, cmap='gray')
        # plt.show()
        return combined_image


## 필터 값 모음
bilateralValues = [[90, 40, 100],
                   [50, 30, 80],
                   [70, 50, 100],
                   [90, 30, 90],
                   [80, 40, 80],
                   [70, 50, 80]]

adaptiveValues = [[91, 13],
                  [91, 17]]


def imageProcessing(original_image, ai_image):
    ## 이미지 로드
    src = cv2.imread(original_image, cv2.IMREAD_ANYCOLOR)
    origin = src.copy()
    src = cv2.resize(origin, dsize=(1024, 512), interpolation=cv2.INTER_AREA)

    bestPercent_AI, bestPercent_CV = 0, 0
    bestResultImageAI = src.copy()

    for t in range(len(bilateralValues)):
        # 1. grayscale
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 2. bilateral filter
        bValue1, bValue2, bValue3 = bilateralValues[t][0], bilateralValues[t][1], bilateralValues[t][2]
        dst_bilateral = cv2.bilateralFilter(src_gray, bValue1, bValue2, bValue3)

        # 3. adaptive threshold
        atValue1, atValue2 = adaptiveValues[0][0], adaptiveValues[0][1]
        dst_adaptiveThreshold = cv2.adaptiveThreshold(dst_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                      atValue1, atValue2)

        # 4. closing - dilation
        open_kernel = np.ones((1, 1), np.uint8)
        close_kernel = np.ones((1, 1), np.uint8)
        dilation = cv2.dilate(dst_adaptiveThreshold, open_kernel, iterations=1)
        dilation = cv2.erode(dilation, close_kernel, iterations=1)

        # 5. median blur
        median = cv2.medianBlur(dilation, 3)

        # cv output
        filterResult = median
        cv_result = src.copy()
        #     cv2.imshow("sample", filterResult)
        for i in range(len(filterResult)):
            for j in range(len(filterResult[0])):
                if filterResult[i][j] == 0:
                    cv_result[i][j][0], cv_result[i][j][1], cv_result[i][j][2] = 0, 0, 255

        # AI image load & grayscale
        src_ai = ai_image
        src_ai = cv2.resize(src_ai, dsize=(1024, 512), interpolation=cv2.INTER_AREA)

        ret, src_ai_threshold = cv2.threshold(src_ai, 0, 255, cv2.THRESH_BINARY)

        # AND 연산
        and_result = src.copy()
        overlapped, AI_found, CV_found = 0, 0, 0

        for i in range(len(src_ai_threshold)):
            for j in range(len(src_ai_threshold[0])):
                if src_ai_threshold[i][j].all() == 0 and filterResult[i][j] == 0:
                    overlapped += 1
                    and_result[i][j][0], and_result[i][j][1], and_result[i][j][2] = 0, 0, 255
                if src_ai_threshold[i][j].all() == 0:
                    AI_found += 1
                if filterResult[i][j] == 0:
                    CV_found += 1

        overlapped_percent_AI = overlapped / AI_found * 100
        overlapped_percent_CV = overlapped / CV_found * 100

        # print("overlaps - AI", round(overlapped_percent_AI, 3), " CV", round(overlapped_percent_CV, 3))
        if bestPercent_AI < overlapped_percent_AI:
            bestPercent_AI = overlapped_percent_AI
            bestResultImageAI = and_result.copy()
            # print("AI - updated")
        print(f'{t+1}/{len(bilateralValues)}')

    print("best intersection - AI", round(bestPercent_AI, 3))

    ## 이미지 저장
    # cv2.imwrite(IMAGE_OUTPUT, bestResultImageAI)
    return bestResultImageAI
