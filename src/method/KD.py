from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilKL(nn.Module):
    """Distil knowlegde in neural network"""
    def __init__(self,T:float):
        super(DistilKL, self).__init__()
        self.T = T
    def forward(self, y_s:torch.Tensor, y_t:torch.Tensor):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean')/y_s.shape[0]
        return loss
    
    
if __name__ == '__main__':
    y_s = torch.randn(2, 10)
    y_t = torch.randn(2, 10)
    loss = DistilKL(10)
    print(loss(y_s, y_t))