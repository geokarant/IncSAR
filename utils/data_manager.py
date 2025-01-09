import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import mstar,mstar_opensar,aircraft,aircraft_mstar_opensar
import torch
from backbone.linears import BEAR_WWTY, BEAR_ABY
import os
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        self._device = args["device"][0]
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, rpca = False, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, rpca, self.rank, self._device)
        else:
            return DummyDataset(data, targets, trsf, self.use_path, rpca, self.args["init_cls"], self.args["dataset"], self._device)


    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, rpca = False, init_cls = None, dataset=None, device= None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.rpca= rpca
        if self.rpca == True:
            if init_cls == 4 :
                if dataset == "aircraft" or dataset=="aircraft_mstar_opensar":
                    path = "./models_rpca/bear_wwty_l0.0001_e80_r2.pth"
                    self.model_rpca = BEAR_WWTY(16384, k = 2)
                else:
                    path = "./models_rpca/bear_rpca_b4inc1.pth"
                    self.model_rpca = BEAR_ABY(16384, k = 10)
            elif init_cls == 2:
                path = "./models_rpca/bear_rpca_b2inc2.pth"
                self.model_rpca = BEAR_ABY(16384, k = 15)
            self.model_rpca.load_state_dict(torch.load(path, map_location=device))
            self.model_rpca.eval()
        else:
            self.model_rpca = None
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx], self.rpca, self.model_rpca))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "mstar":
        return mstar(args["backbone_type"],args["setup"],args["portion"])
    elif name == "mstar_opensar":
        return mstar_opensar(args["backbone_type"])
    elif name == "aircraft":
        return aircraft(args["backbone_type"],args["setup"],args["portion"])
    elif name == "aircraft_mstar_opensar":
        return aircraft_mstar_opensar(args["backbone_type"],args["setup"],args["portion"])
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
    

def pil_loader(path, rpca, model_rpca):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    center_crop = transforms.CenterCrop((128, 128))
    with open(path, "rb") as f:
        if rpca == False:
            img = Image.open(f)
            img_rgb = img.convert("RGB")
        else: 
            img = Image.open(f).convert('L')  
            img_resized = center_crop(img)  # Resize the image to the target size
            img_np = np.array(img_resized).astype(float)
            Y = torch.from_numpy(img_np).float()
            Y_res = Y.view(1, -1)            
            L = model_rpca(Y_res)
            S = Y_res - L
            S = S.view(128,128)
            S_np = S.detach().cpu().numpy()
            img_rgb = Image.fromarray(S_np)  # Create a PIL Image from NumPy array
            img_rgb = img_rgb.convert("RGB")

        return img_rgb
