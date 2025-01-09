import os
import numpy as np
import torch
from torch.utils.data import Dataset

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}

    # Total accuracy
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    # For incremental classes
    for class_id in range(init_cls, np.max(y_true) + 1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    """
    # Accuracy for each class separately
    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        idxes = np.where(y_true == cls)[0]
        all_acc[str(cls)] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    """
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def soft_voting(logits1, logits2, y_true , init_cls, increment):
    import torch.nn.functional as F

    size = logits1.size()
    known_classes = size[1]
    
    probabilities_1 = F.softmax(logits1, dim=1)
    probabilities_2 = F.softmax(logits2, dim=1)

    combined_predictions = (probabilities_1 + probabilities_2) / 2 

    predicts = torch.topk(combined_predictions, k=1, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    predicts= predicts.cpu().numpy()
    ret = {}
    grouped = accuracy(predicts.T[0], y_true, known_classes-1, init_cls, increment)
    ret["grouped"] = grouped
    ret["top1"] = grouped["total"]
    ret["top{}".format(1)] = np.around(
            (predicts.T == np.tile(y_true, (1, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
    return ret


def soft_voting_analytical(logits1, logits2, y_true , init_cls, increment):
    import torch.nn.functional as F

    size = logits1.size()
    known_classes = size[1]
    
    probabilities_1 = F.softmax(logits1, dim=1)
    probabilities_2 = F.softmax(logits2, dim=1)

    combined_predictions = (probabilities_1 + probabilities_2) / 2 

    predicts = torch.topk(combined_predictions, k=1, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    predicts= predicts.cpu().numpy()
    ret = {}
    grouped = accuracy(predicts.T[0], y_true, known_classes-1, init_cls, increment)
    ret["grouped"] = grouped
    ret["top1"] = grouped["total"]
    ret["top{}".format(1)] = np.around(
            (predicts.T == np.tile(y_true, (1, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
    
    predicts_vit = torch.topk(logits1, k=1, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    predicts_vit= predicts_vit.cpu().numpy()
    ret_vit = {}
    grouped_vit = accuracy(predicts_vit.T[0], y_true, known_classes-1, init_cls, increment)
    ret_vit["grouped"] = grouped_vit
    ret_vit["top1"] = grouped_vit["total"]
    ret_vit["top{}".format(1)] = np.around(
            (predicts_vit.T == np.tile(y_true, (1, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
    
    predicts_cnn = torch.topk(logits2, k=1, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    predicts_cnn= predicts_cnn.cpu().numpy()

    ret_cnn = {}
    grouped_cnn = accuracy(predicts_cnn.T[0], y_true, known_classes-1, init_cls, increment)
    ret_cnn["grouped"] = grouped_cnn
    ret_cnn["top1"] = grouped_cnn["total"]
    ret_cnn["top{}".format(1)] = np.around(
            (predicts_cnn.T == np.tile(y_true, (1, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
    

    return ret, ret_vit, ret_cnn

class PairedDataset(Dataset):
    #Dataset class to pair ViT and CNN images along with the corresponding label
    def __init__(self, vit_dataset, cnn_dataset):
        assert len(vit_dataset) == len(cnn_dataset), "ViT and CNN datasets must have the same length!"
        self.vit_dataset = vit_dataset
        self.cnn_dataset = cnn_dataset

    def __len__(self):
        return len(self.vit_dataset)

    def __getitem__(self, idx):
        # Get the ViT and CNN images and their labels
        _, image_vit, label_vit = self.vit_dataset[idx]
        _, image_cnn, label_cnn = self.cnn_dataset[idx]
        
        # Ensure the labels match between the two datasets
        assert label_vit == label_cnn, "Labels from ViT and CNN datasets do not match!"
        
        # Return the paired images and the label
        return image_vit, image_cnn, label_vit