

import torch
import torch.nn as nn

class Similarity(nn.Module):
    """Similarity-Peserving knowledge distillation, ICCV2019"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s,g_t):
        return [self.similarity(f_s,f_t) for f_s,f_t in zip(g_s,g_t)] 
    
    def similarity(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = f_s @ f_s.T
        G_s = G_s / G_s.norm(dim=1, keepdim=True)

        G_t = f_t @ f_t.T
        G_t = G_t / G_t.norm(dim=1, keepdim=True)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

if __name__ == '__main__':
    y_s = torch.randn(2, 10)
    y_t = torch.randn(2, 10)
    loss = Similarity()
    print(y_s,y_t)
    print(loss(y_s, y_t))