import os
import torch
import numpy as np
from PIL import Image
# 데이터 로더를 구현하기

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_input.sort()

        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        input = Image.open(os.path.join(self.data_dir, self.lst_input[index]))
        input = np.array(input)

        # 정규화
        input = input/255.0

        # 이미지의 차원 = 2일 경우(채널이 없을 경우, 흑백 이미지), 새로운 채널(축) 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input}

        # transform이 정의되어 있다면 transform을 거친 데이터를 불러옴
        if self.transform:
            data = self.transform(data)

        return data
