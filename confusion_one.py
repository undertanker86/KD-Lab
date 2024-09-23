from src.model import resnet18one

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import cal_param_size, cal_multi_adds

# from torchmetrics import ConfusionMatrix
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    accuracy_score,
)
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, normalize, resize
# from torchcam.methods import GradCAM
# from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import numpy as np


def collect_predictions(dataloader, model):
    all_labels = []
    all_preds = {"out1": [], "out2": [], "out3": [], "out4": [], "x_m": []}

    with torch.no_grad():
        for images, labels in dataloader:
            [out4, out3, out2, out1, x_m], _ = model(images)
            outputs = {
                "out1": out1,
                "out2": out2,
                "out3": out3,
                "out4": out4,
                "x_m": x_m,
            }
            all_labels.extend(labels.cpu().numpy())
            for key in outputs:
                _, preds = torch.max(outputs[key], 1)
                all_preds[key].extend(preds.cpu().numpy())

    return all_labels, all_preds


def normalize_confusion_matrix(cm):
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(
        cm_normalized
    )  # Replace NaNs with zero if any row sums are zero
    return cm_normalized


def calculate_top1_error_rate(preds, labels):
    correct = np.sum(np.array(preds) == np.array(labels))
    total = len(labels)
    top1_error_rate = 1 - correct / total
    return top1_error_rate


def main():
    model = resnet18one()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(
        "D:/Code-AMinh-Conf/Self-distil/resnet18-epoch106.pth")
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    classnames = ["surprise", "fear", "disgust",
                  "happy", "sad", "angry", "neutral"]
    # classnames = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    # from utils import cal_multi_adds, cal_param_size

    print(
        "Params: %.2fM, Multi-adds: %.3fM"
        % (cal_param_size(model) / 1e6, cal_multi_adds(model, (2, 3, 32, 32)) / 1e6)
    )
    target_size = 48
    # mean = [0.5756, 0.4495, 0.4010]
    # std = [0.2599, 0.2365, 0.2351]
    mean = 48
    std = 255
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Load the dataset
    data_dir = "D:/Code-AMinh-Conf/Self-distil/rafdb/DATASET/test"
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=False, num_workers=4)

    # Collect predictions
    all_labels, all_preds = collect_predictions(dataloader, model)

    # Generate and visualize confusion matrices for each output
    for key in all_preds:
        cm = confusion_matrix(all_labels, all_preds[key])
        cm_normalized = normalize_confusion_matrix(cm)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, display_labels=classnames
        )

        f1 = f1_score(all_labels, all_preds[key], average="weighted")
        recall = recall_score(all_labels, all_preds[key], average="weighted")
        precision = precision_score(
            all_labels, all_preds[key], average="weighted")
        top1_error_rate = calculate_top1_error_rate(all_preds[key], all_labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(
            ax=ax, cmap="Blues", values_format=".2f"
        )  # 'Blues' colormap and format values to 2 decimal places
        plt.title(f"Normalized Confusion Matrix for {key}")
        plt.savefig(f"images/raf/confusion_matrix_{key}.png")
        plt.show()

        print(f"Metrics for {key}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Top-1 Error Rate: {top1_error_rate:.4f}\n")

        # Calculate and print classification report
        report = classification_report(
            all_labels, all_preds[key], target_names=classnames
        )
        # df = pandas.DataFrame(report).transpose()
        # df.to_csv(f"images/confusion+/classification_report_{key}.csv")

        print(f"Classification Report for {key}:\n{report}")

        # Calculate overall accuracy
        accuracy = accuracy_score(all_labels, all_preds[key])
        print(f"Overall Accuracy for {key}: {accuracy:.4f}")

        # Calculate top-1 error rate
        top1_error_rate = calculate_top1_error_rate(all_preds[key], all_labels)
        print(f"Top-1 Error Rate for {key}: {top1_error_rate:.4f}\n")


# def grad_cam(model, input_tensor, class_index, target_layer="model.backbone.layer4"):
#     with GradCAM(model, target_layer=target_layer) as cam_extractor:
#         # [out4, out3, out2, out1, x_m] as fc output
#         out = model(input_tensor)[0][0]
#         activation_map = cam_extractor(class_index, out)
#         return activation_map


# def display_cam_overlay(image, activation_map):
#     image = image.squeeze(0)
#     result = overlay_mask(
#         to_pil_image(image),
#         to_pil_image(activation_map[0].squeeze(0), mode="F"),
#         alpha=0.5,
#     )
#     plt.imshow(result)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig("gradcam.png", dpi=300)
#     plt.show()


if __name__ == "__main__":
    main()
    # model = resnet18one()
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('ckpt/resnetone/resnet18-epoch180.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # model.eval()
    # # print(model.layer4)
    # target_size = 48
    # mean = 0
    # std = 255
    # image = read_image('test/disgust/PrivateTest_807646.jpg')
    # transform =transforms.Compose([
    #     transforms.Resize((target_size, target_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    # # Load the dataset
    # data_dir = 'test'
    # dataset = ImageFolder(root=data_dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # input_tensor, label = next(iter(dataloader))
    # # display image tensor in plt
    # # print(label)
    # # Generate Grad-CAM
    # activation_map = grad_cam(model,input_tensor, label.item(),target_layer="layer4")
    # display_cam_overlay(input_tensor, activation_map)
