from src.model import resnet18one

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cal_param_size, cal_multi_adds
# from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

def collect_predictions(dataloader, model):
    all_labels = []
    all_preds = {'out1': [], 'out2': [], 'out3': [], 'out4': [], 'x_m': []}

    with torch.no_grad():
        for images, labels in dataloader:
            [out4, out3, out2, out1, x_m],_ = model(images)
            outputs = {'out1': out1, 'out2': out2, 'out3': out3, 'out4': out4, 'x_m': x_m}
            all_labels.extend(labels.cpu().numpy())
            for key in outputs:
                _, preds = torch.max(outputs[key], 1)
                all_preds[key].extend(preds.cpu().numpy())
    
    return all_labels, all_preds
def main():
    model = resnet18one()
    model_dict = model.state_dict()
    pretrained_dict = torch.load('ckpt/resnetone/resnet18-epoch180.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()

    from utils import cal_multi_adds, cal_param_size
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(model) / 1e6, cal_multi_adds(model, (2, 3, 32, 32)) / 1e6))
    target_size = 48
    mean = 0
    std = 255
    transform =transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load the dataset
    data_dir = 'test'
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)



# Collect predictions
    all_labels, all_preds = collect_predictions(dataloader, model)

    # Generate and visualize confusion matrices for each output
    for key in all_preds:
        cm = confusion_matrix(all_labels, all_preds[key])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)
        plt.title(f'Confusion Matrix for {key}')
        
        plt.show()



if __name__ == '__main__':
    main()