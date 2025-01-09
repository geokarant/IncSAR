import copy
import logging
import numpy as np
import torch
from torch import nn
from utils.toolkit import tensor2numpy, accuracy

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args


    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def eval_task_attention_fusion(self):
        y_pred, y_true = self._eval_cnn_attention_fusion(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def _eval_cnn_attention_fusion(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for i, (data_vit, data_cnn, targets) in enumerate(loader):

            data_vit = data_vit.to(self._device)
            data_cnn = data_cnn.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                outputs = self._network(data_vit,data_cnn)["logits"]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    
    def eval_task_late_fusion(self):
        logits, y_pred, y_true = self._eval_cnn_late_fusion(self.test_loader)
        return logits, y_pred, y_true
    def _eval_cnn_late_fusion(self, loader):
        self._network.eval()
        y_pred, y_true, outputs = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                output = self._network(inputs)["logits"]
            predicts = torch.topk(output, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            outputs.append(output)
        outputs_total = torch.cat(outputs, dim=0)
        return outputs_total, np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _compute_accuracy_attention_fusion(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, ( data_vit, data_cnn, targets) in enumerate(loader):

            data_vit = data_vit.to(self._device)
            data_cnn = data_cnn.to(self._device)
            targets = targets.to(self._device) 
            with torch.no_grad():
                outputs = model(data_vit,data_cnn)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets.cpu()).sum()
            total += len(targets.cpu())

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
