import os
import torch

## 네트워크 불러오기
def load(ckpt_dir, net):
    if not os.path.exists(ckpt_dir):
        print('warning! there is no checkpoint ')
        return net
    
    ckpt_path = os.path.join(ckpt_dir,'model_epoch60.pth')
    
    dict_model = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(dict_model['net'])
    
    return net