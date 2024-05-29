import torch, torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import torch.optim as optim, torch.nn as nn, torch.nn.functional as F, torch.utils.data as data
import os, random, copy, pdb, numpy as np, ssl
from PIL import Image
from scipy.spatial.distance import cdist
from numpy import loadtxt

ssl._create_default_https_context = ssl._create_unverified_context  # fix needed for downloading resnet weights
device = torch.device('cuda:0')  # gpu training is typically much faster if available


# device = torch.device('cpu')

# read the dataset files and return the list of images and list of class labels
def readCUB(kept_labels, n=1000):
    cub_folder = 'CUB_200_2011/'
    f = open(cub_folder + "images.txt", 'r')
    cub_imgfn = [a.split(' ')[::-1] for a in f.read().split('\n')]
    cub_label = loadtxt(cub_folder + "image_class_labels.txt", delimiter=" ", unpack=False)[:, 1].astype(int) - 1
    idx = np.isin(cub_label, kept_labels).nonzero()[0]
    cub_imgfn = [cub_imgfn[x] for x in idx]
    cub_label = cub_label[idx]

    idx = []
    for i in range(0, max(cub_label) + 1):
        idx = idx + ([x for (x, val) in enumerate(cub_label) if val == i][0:n])
    cub_imgfn = [cub_imgfn[x] for x in idx]
    cub_label = cub_label[idx]

    cub_imgfn = [(cub_folder + '/images/' + x[0]) for x in cub_imgfn]
    cub_label = np.array(cub_label)

    return cub_imgfn, cub_label


# dataset structure used for testing/evaluation
class CUB(data.Dataset):

    def __init__(self, kept_labels, transform=None, n=1000):
        self.img, self.labels = readCUB(kept_labels, n)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert('RGB')
        label = self.labels[index].item()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img)


# dataset structure used for training
class CUBtriplet(data.Dataset):

    def __init__(self, kept_labels, transform=None, n=1000):

        self.img, self.labels = readCUB(kept_labels, n)
        self.transform = transform
        self.class_idx = dict()
        # for l in kept_labels: self.class_idx[l] = np.where(self.labels==l)[0] # indices of images for every class
        self.hardneg = np.zeros(self.__len__(), np.uint32)

    # this is used to iterate over images and extract descriptors for hard-negative mining        
    def getimg(self, index):
        img = Image.open(self.img[index]).convert('RGB')
        img = self.transform(img)
        label = self.labels[index].item()
        return img, label

    # extract the representation of all training images
    def extract_images(self, model):
        model.eval()
        des = torch.Tensor(self.__len__(), model.dim)
        labels = torch.LongTensor(self.__len__())
        for i in range(self.__len__()):
            (data, target) = self.getimg(i)
            with torch.no_grad():
                des[i] = model(data.to(device).unsqueeze(0)).cpu()
            labels[i] = target
        return des, labels

    def minehard(self, model):
        descriptors, labels = self.extract_images(model)
        for i, (descriptor, label) in enumerate(zip(descriptors, labels)):
            dists = torch.cdist(descriptor.unsqueeze(0), descriptors).flatten()
            argsort = torch.argsort(dists)
            argsort_idx = copy.deepcopy(i)
            while labels[argsort_idx] == label:
                idx = torch.randint(low=0, high=30, size=(1,)).item()
                argsort_idx = argsort[idx]

            self.hardneg[i] = argsort_idx

            # import matplotlib.pyplot as plt
            # img, img_pos, img_neg, lab1, lab2, lab3 = self.__getitem__(i, True)
            # fig, axs = plt.subplots(1, 3)
            # axs[0].set_title(f'Img class: {labels[i]} {lab1}')
            # axs[0].imshow(img.detach().numpy().squeeze().transpose(1,2,0))
            # axs[1].set_title(f'Img Pos {lab2}')
            # axs[1].imshow(img_pos.detach().numpy().squeeze().transpose(1,2,0))
            # axs[2].set_title(f'Img Neg {lab3}')
            # axs[2].imshow(img_neg.detach().numpy().squeeze().transpose(1,2,0))
            # plt.tight_layout()
            # plt.show()
        # mine hard negatives and store them in "self.hardneg" - your code
        # ........
        # ........
        # ........
        # ........
        # ........

    # this is used to iterate over triplets 
    # and is called by the data-loader for batch construction
    def __getitem__(self, index, return_labels=False):
        img1, lab1 = Image.open(self.img[index]).convert('RGB'), self.labels[index].item()
        img1 = self.transform(img1)

        positive_mask = self.labels == lab1
        rand_pos_idx = np.random.randint(low=0, high=np.sum(positive_mask).item(), size=(1,)).item()
        idxs_2 = np.argwhere(positive_mask)
        idx2 = idxs_2[rand_pos_idx].item()
        while idx2 == index:
            rand_pos_idx = np.random.randint(low=0, high=np.sum(positive_mask).item(), size=(1,)).item()
            idx2 = idxs_2[rand_pos_idx].item()
        img2, lab2 = self.getimg(idx2)

        img3, lab3 = self.getimg(self.hardneg[index])
        # img1 is the anchor, pick a positive and a hard negative - your code
        # ........
        # ........
        # ........
        # ........
        # ........
        if return_labels:
            return img1, img2, img3, lab1, lab2, lab3
        return img1, img2, img3

    # this lets the data-loader know how many items are there to use
    # note that number of triplets = number of training images, since we are using each image once as an anchor
    def __len__(self):
        return len(self.img)


class GDextractor(nn.Module):
    """
    Create A network that maps an image to an embedding (descriptor) with global pooling
    """

    def __init__(self, input_net, dim, usemax=False):
        """
        Contructor takes a CNN as input
        input_net: a ResNet
        dim: output embedding dimensionality
        usemax: do MAC is true, otherwise SPoC
        """
        super(GDextractor, self).__init__()
        self.dim = dim
        self.usemax = usemax

        # create the network structure by using part of input_net - your code
        # ........
        # ........
        # ........
        self.part1 = nn.Sequential(*list(input_net.children())[:-2])
        if self.usemax:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.part2 = nn.Sequential(nn.Flatten(), nn.Linear(512, dim))
        self.part3 = nn.functional.normalize

    def forward(self, x, eps=1e-6):
        # x are the input image in the batch, extract the global descriptor - your code
        # ........
        # ........
        # ........
        # ........
        # z = .... # global descriptor
        z = self.part1(x)
        z = self.pool(z)
        z = self.part2(z)
        z = self.part3(z, eps=eps,)
        return z


def test(model, test_loader):
    """
    Compute accuracy on the test set
    model: network
    test_loader: test_loader loading images and labels in batches
    """

    model.eval()
    des = torch.Tensor()
    labels = torch.LongTensor()
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            des = torch.cat((des, model(data.to(device)).cpu()))
        labels = torch.cat((labels, target))

    # compute all pair-wise distances
    cdistances = cdist(des.data.numpy(), des.data.numpy(), 'euclidean')

    # sort distances and see if the label of the top-ranked image is correct
    prec = np.zeros(len(cdistances))
    for i in range(0, len(cdistances)):
        idx = np.argsort(cdistances[i])[1]  # skip 1st image - is the query itself
        prec[i] = ((labels[idx] == labels[i]) * 1.0).mean().item()

    return prec.mean()


def triplet_loss(distances_pos, distances_neg, margin):
    # input: pos. and neg. distances per triplet in the batch
    # compute and return the loss per triplet - your code
    # ........
    loss = torch.nn.ReLU()(distances_pos - distances_neg + margin)
    return loss


def train(model, train_loader, optimizer, margin=0.5):
    """
    Training of an epoch with Triplet loss and triplets with hard-negatives
    model: network
    train_loader: train_loader loading triplets of the form (a,p,n) in batches. 
    optimizer: optimizer to use in the training
    margin: triplet loss margin
    """

    model.train()  # first put the model intro training mode
    model.apply(
        set_batchnorm_eval)  # do not update the Batch-Norm running avg/std - helps to get improvements in a couple of epochs
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):  # iterate over batches
        # call the model and get global descriptors v1,v2,v3 for all anchors, positives and negatives in the batch
        v1, v2, v3 = model(data[0].to(device)), model(data[1].to(device)), model(data[2].to(device))

        # compute the required distances - your code
        # ........
        # ........

        distances_pos = (v2 - v1).pow(2).sum(1)
        distances_neg = (v3 - v1).pow(2).sum(1)
        loss = triplet_loss(distances_pos, distances_neg, margin)

        # update the network with back-propagation - your code
        # ........
        # ........
        loss.sum().backward()
        optimizer.step()

        total_loss = total_loss + loss.mean().cpu().item()
        print(total_loss)

    print('Epoch average loss {:.6f}'.format(total_loss / batch_idx))


# sets the BN layers to eval mode
# so that the running statistics will not get updated
def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main():
    ## input transformations for training (image augmentations) and testing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # these hyper-parameters worked reasonably well for us
    lr = 0.00001
    margin = .1

    trainset = CUBtriplet(kept_labels=np.arange(100), n=20, transform=transform_train)  # keep only 20 images per class
    valset = CUB(kept_labels=np.arange(100, 150), transform=transform_test)

    torch.manual_seed(0);
    np.random.seed(0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)  # 64 triplets per batch
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)

    pre_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)  # load pre-trained ResNet18
    dim = 512  ## dimensionality of the global descriptor - defined according to the ResNet18 architecture
    model = GDextractor(pre_model, dim)  # construct the network that extracts descriptors
    model.to(device)

    best_mp = test(model, valloader)
    torch.save({'epoch': 0, 'val_mp': best_mp, 'state_dict': model.state_dict()}, 'bestmodel.pth.tar')
    print('Before training, precision@1: {}'.format(best_mp), flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print('Training...')
    for epoch in range(1, 10 + 1):
        trainset.minehard(model)
        print('Epoch {}'.format(epoch), flush=True)
        train(model, trainloader, optimizer, margin)

        if epoch % 1 == 0:
            mp = test(model, valloader)
            print('Epoch {}, precision@1: {}'.format(epoch, mp), flush=True)

            if mp > best_mp:
                best_mp = mp
                torch.save({'epoch': epoch, 'val_mp': mp, 'state_dict': model.state_dict()}, f'bestmodel_{epoch}_alt.pth.tar')


if __name__ == '__main__':
    main()
