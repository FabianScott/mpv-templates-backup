import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List, Dict
from PIL import Image
import os

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from time import time


def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    '''Function, that calculates mean and std of a dataset (pixelwise)
    Return:
        tuple of Lists of floats. len of each list should equal to number of input image/tensor channels
    '''
    mean = torch.tensor([0., 0., 0.], device=dataset[0][0].device)
    std = torch.tensor([0., 0., 0.], device=dataset[0][0].device)
    length = len(dataset)
    for img, label in dataset:
        mean += torch.mean(img, dim=[1, 2], keepdim=True).flatten()
        std += torch.std(img, dim=[1, 2], keepdim=True).flatten()

    return [el.item() for el in mean / length], [el.item() for el in std / length]


class SimpleCNN(nn.Module):
    """Class, which implements image classifier. """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),  # Depthwise convolution
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),  # Pointwise convolution
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),  # Dilated convolution
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
        )
        self.clf = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),
                                 nn.Linear(512, num_classes))
        return

    def forward(self, input):
        """ 
        Shape:
        - Input :math:`(B, C, H, W)` 
        - Output: :math:`(B, NC)`, where NC is num_classes
        """
        x = self.features(input)
        return self.clf(x)


def weight_init(m: nn.Module) -> None:
    '''Function, which fills-in weights and biases for convolutional and linear layers'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    return


def train_and_val_single_epoch(model: torch.nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               val_loader: torch.utils.data.DataLoader,
                               optim: torch.optim.Optimizer,
                               loss_fn: torch.nn.Module,
                               epoch_idx=0,
                               lr_scheduler=None,
                               writer=None,
                               device: torch.device = torch.device('cpu'),
                               additional_params: Dict = {}) -> torch.nn.Module:
    '''Function, which runs training over a single epoch in the dataloader and returns the model. Do not forget to set the model into train mode and zero_grad() optimizer before backward.'''
    do_acc = True
    if epoch_idx == 0:
        val_loss, additional_out = validate(model, val_loader, loss_fn, device, additional_params)
        model = model.to(device)
        if writer is not None:
            if do_acc:
                writer.add_scalar("Accuracy/val", additional_out['acc'], 0)
                print(f'Validation Accuracy: {additional_out["acc"]}')
            writer.add_scalar("Loss/val", val_loss, 0)
    model.train()
    for idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
        loss, acc = train_step(model, data, labels, loss_fn, optim, do_acc=do_acc)
        lr_scheduler.step() if lr_scheduler is not None else None
        print('Loss: ', loss.item())
        if do_acc:
            print('Training Accuracy: ', acc.item())

    return model


def lr_find(model: torch.nn.Module,
            train_dl: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            min_lr: float = 1e-7, max_lr: float = 100, steps: int = 50) -> Tuple:
    '''Function, which run the training for a small number of iterations, increasing the learning rate and storing the losses.
    Model initialization is saved before training and restored after training'''
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), steps)
    losses = np.ones(steps)
    t = time()
    small_dl = []
    for idx, (data, labels) in tqdm(enumerate(train_dl), total=2, desc='Getting data'):
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
        small_dl.append((data, labels))
        if idx == 2:
            print(f'Data loaded: {time() - t:.3} seconds')
            break

    for i, lr in tqdm(enumerate(lrs), desc='Finding LR', total=len(lrs)):
        model_copy = copy.deepcopy(model)
        optim = torch.optim.Adam(model_copy.parameters(), lr=lr)
        for idx, (data, labels) in enumerate(small_dl):
            loss, _ = train_step(model_copy, data, labels, loss_fn, optim)
            print(f'Loss at lr={lr}: {loss.item()}')
            losses[i] = loss

    return losses, lrs


def validate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device = torch.device('cpu'),
             additional_params: Dict = {}) -> Tuple[float, Dict]:
    '''Function, which runs the module over validation set and returns accuracy'''
    print("Starting validation")
    acc = []
    loss = 0
    do_acc = True
    model.to(device)
    model.eval()
    if 'with_acc' in additional_params:
        do_acc = additional_params['with_acc']
    for idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            preds = model(data)
            loss += loss_fn(preds, labels)
            if do_acc:
                acc.append(torch.sum(preds.argmax(dim=1) == labels) / len(labels))

    return loss, {'acc': torch.mean(torch.tensor(acc))}


class TestFolderDataset(torch.utils.data.Dataset):
    '''Class, which reads images in folder and serves as test dataset'''

    def __init__(self, folder_name, transform=None):
        self.dataset = []
        self.fnames = []
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                img = Image.open(os.path.join(root, file))
                if transform is not None:
                    img = transform(img)
                img_tensor = torch.tensor(np.array(img), dtype=torch.float)

                self.dataset.append(img_tensor)
                self.fnames.append(file)
        self.dataset = torch.stack(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]
        return img

    def __len__(self):
        return len(self.dataset)


def get_predictions(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
    '''Function, which predicts class indexes for image in data loader. Ouput shape: [N, 1], where N is number of image in the dataset'''
    out = model(test_dl.dataset.dataset).argmax(1)
    return out


def train_step(model: torch.nn.Module, data, labels, loss_fn, optim, do_acc=False):
    optim.zero_grad()
    preds = model(data)
    loss = loss_fn(preds, labels)
    loss.backward()
    optim.step()
    if do_acc:
        acc = torch.sum(preds.argmax(1) == labels) / len(labels)
        return loss, acc
    return loss, None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as np
    import cv2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision as tv
    import kornia as K
    from tqdm import tqdm_notebook as tqdm
    from time import time

    torch.set_default_device('cuda:0')


    def imshow_torch(tensor, figsize=(8, 6), *kwargs):
        plt.figure(figsize=figsize)
        plt.imshow(K.tensor_to_image(tensor), *kwargs)
        return


    def imshow_torch_channels(tensor, dim=1, *kwargs):
        num_ch = tensor.size(dim)
        fig = plt.figure(figsize=(num_ch * 5, 5))
        tensor_splitted = torch.split(tensor, 1, dim=dim)
        for i in range(num_ch):
            fig.add_subplot(1, num_ch, i + 1)
            plt.imshow(K.tensor_to_image(tensor_splitted[i].squeeze(dim)), *kwargs)
        return

    import torchvision.transforms as tfms

    train_transform = tfms.Compose([tfms.Resize((128, 128)),
                                    tfms.ToTensor()])

    ImageNette_for_statistics = tv.datasets.ImageFolder('imagenette2-160/train',
                                                        transform=train_transform, )

    mean, std = get_dataset_statistics(ImageNette_for_statistics)
    for i in range(3):
        imshow_torch(ImageNette_for_statistics[i][0], figsize=(3, 3))

    train_transform = tfms.Compose([tfms.Resize((128, 128)),
                                    tfms.RandomHorizontalFlip(),
                                    tfms.ToTensor(),
                                    tfms.Normalize(mean, std)])

    val_transform = tfms.Compose([tfms.Resize((128, 128)),
                                  tfms.ToTensor(),
                                  tfms.Normalize(mean, std)])

    ImageNette_train = tv.datasets.ImageFolder('imagenette2-160/train',
                                               transform=train_transform)

    ImageNette_val = tv.datasets.ImageFolder('imagenette2-160/val',
                                             transform=val_transform)
    import os

    num_workers = os.cpu_count()
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2
    print(num_workers)

    BS = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dl = torch.utils.data.DataLoader(ImageNette_train,
                                           batch_size=32,
                                           shuffle=True,  # important thing to do for training.
                                           num_workers=num_workers,
                                           generator=torch.Generator(device=device))
    val_dl = torch.utils.data.DataLoader(ImageNette_val,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         drop_last=False,
                                         generator=torch.Generator(device=device))
    model = SimpleCNN(10)
    model.features.apply(weight_init)
    model.clf.apply(weight_init)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    start_epoch = 14
    writer = SummaryWriter(comment="AdamW_OneCycleSched")

    for ep in range(start_epoch, start_epoch + epochs):
        model_filename = f'models/fromscript_at_{ep}'
        if ep == start_epoch:
            try:
                model = torch.load(model_filename)
                print(f'Loaded model from {model_filename}')
            except FileNotFoundError:
                print(f'No model found at {model_filename}')
                pass
            opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-2)
            lr_sched = OneCycleLR(opt1, max_lr=3e-3, steps_per_epoch=len(train_dl), epochs=epochs)

        t = time()
        model = train_and_val_single_epoch(model, train_dl, val_dl, opt1, loss_fn, 0, lr_sched, writer=writer,
                                           device=torch.device('cuda'), additional_params={"with_acc": True})
        el = time() - t
        print(f"Train epoch in {el:.1f} sec")
        torch.save(model, model_filename)

