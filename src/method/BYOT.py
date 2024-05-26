import torch 
import torch.nn as nn
import torch.nn.functional as F

import math


def BYOT(model, inputs, targets,logit_loss, beta_loss):
    loss_div = torch.tensor(0.0).to(inputs.device)# kl loss
    loss_cls = torch.tensor(0.0).to(inputs.device)# ce loss
    logits, feature = model(inputs, feature=True)
    for i in range(len(logits)):
        loss_cls += logit_loss(logits[i],targets)
        if i != 0:
            loss_div += beta_loss(logits[i], logits[0].detach())
    
    
    # feature loss
    for i in range(1, len(feature)):
        if i != 1:
            loss_div += 0.5*0.1*((feature[i]-feature[1].detach())**2).mean()


    logit = logits[0]
    return logit, loss_div, loss_cls