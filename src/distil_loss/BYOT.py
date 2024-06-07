import torch 
import torch.nn as nn
import torch.nn.functional as F

import math

# for paper 20219 use v1 else use v2 to remove feature loss
# Can add other feature loss or contrastive loss support later on
def BYOT(model, inputs, targets,logit_loss, beta_loss,version='v1'):
    loss_div = torch.tensor(0.0).to(inputs.device)# kl loss
    loss_cls = torch.tensor(0.0).to(inputs.device)# ce loss
    logits, feature = model(inputs, feature=True)
    for i in range(len(logits)):
        loss_cls += logit_loss(logits[i],targets)
        if i != 0:
            loss_div += beta_loss(logits[i], logits[0].detach())
    
    if version == 'v1':
    # feature loss
        for i in range(1, len(feature)):
            if i != 1:
                loss_div += 0.5*0.1*((feature[i]-feature[1].detach())**2).mean()


    logit = logits[0]
    return logit, loss_div, loss_cls