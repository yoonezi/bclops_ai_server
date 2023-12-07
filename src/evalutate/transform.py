import torch
import numpy as np

# 트랜스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        input = data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = data['input']

        input = (input - self.mean) / self.std

        data = {'input': input}

        return data
