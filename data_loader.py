from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.io import imsave, imread
import os
from gtsrb import GTSRB

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif (opt.dataset == 'CIFAR100'):
        trainset = datasets.CIFAR100(root='data/CIFAR100', train=True, download=True)
    elif (opt.dataset == 'GTSRB'):
        trainset = GTSRB(train=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    elif (opt.dataset == 'CIFAR100'):
        testset = datasets.CIFAR100(root='data/CIFAR100', train=False, download=True)
    elif (opt.dataset == 'GTSRB'):
        testset = GTSRB(train=False)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif (opt.dataset == 'CIFAR100'):
        trainset = datasets.CIFAR100(root='data/CIFAR100', train=True, download=True)
    elif (opt.dataset == 'GTSRB'):
        trainset = GTSRB(train=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return train_data_bad, train_bad_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label

    def __len__(self):
        return self.dataLen


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda")):
        self.trigger = self.get_trigger(opt.trigger_type)
        self.gnd = full_dataset.targets
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)
        triggered = self.dataset[item][2] # whether the data is triggered/poisoned

        return img, label, triggered

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, trigger_type)

                        # change target
                        dataset_.append((img, target_label, True))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], False))

                elif mode == 'test':                
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    
                    if inject_portion == 0: # clean test set
                        dataset_.append((img, data[1], False))
                    elif inject_portion == 1:
                        if data[1] != target_label:
                            img = self.selectTrigger(img, width, height, trigger_type)
                            dataset_.append((img, target_label, True))
                            cnt += 1

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_, True))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], False))

                elif mode == 'test':

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if inject_portion == 0:
                        dataset_.append((img, data[1], False))
                    elif inject_portion == 1:
            
                        img = self.selectTrigger(img, width, height, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_, True))
                        cnt += 1

            # clean label attack
            elif target_type == 'cleanLabel': # only for trigger_type==sig

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if data[1] == target_label and cnt < int(len(dataset) * inject_portion):
                        img = self.selectTrigger(img, width, height, trigger_type)

                        dataset_.append((img, data[1], True))
                        cnt += 1

                    else:
                        dataset_.append((img, data[1], False))

                elif mode == 'test':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if inject_portion == 0:
                        dataset_.append((img, data[1], False))
                    elif inject_portion == 1:
                        if data[1] != target_label:
                            img = self.selectTrigger(img, width, height, trigger_type)
                            dataset_.append((img, target_label, True))
                            cnt += 1

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, triggerType):

        assert triggerType in ['clean', 'badnet_sq', 'badnet_grid', 'trojan_3x3', 'trojan_8x8', 'trojan_wm', 'l0_inv', 'l2_inv', 'blend', 'smooth', 'sig']

        if triggerType == 'badnet_sq':
            img[32-1-4:32-1, 32-1-4:32-1, :] = 255

        elif triggerType == 'badnet_grid':
            img[width - 1][height - 1] = 255
            img[width - 1][height - 2] = 0
            img[width - 1][height - 3] = 255

            img[width - 2][height - 1] = 0
            img[width - 2][height - 2] = 255
            img[width - 2][height - 3] = 0

            img[width - 3][height - 1] = 255
            img[width - 3][height - 2] = 0
            img[width - 3][height - 3] = 0

        else:
            trigger = self.trigger

            if triggerType == 'blend':
                img = 0.8 * img + 0.2 * trigger
            elif triggerType == 'sig': # grey scale pattern img
                img = 0.8 * img + 0.2 * trigger.reshape((width, height, 1))
            elif triggerType == 'smooth':
                img = img + trigger
                img = normalization(img) * 255
            elif triggerType == 'l0_inv':
                mask = 1 - np.transpose(np.load('./trigger/mask.npy'), (1, 2, 0)) # ndarray, shape=(32, 32, 3)
                img = img * mask + trigger
            elif triggerType == 'clean':
                pass # no trigger is added
            else:
                img = img + trigger

        img = np.clip(img.astype('uint8'), 0, 255)

        return img

    def get_trigger(self, triggerType):
        if triggerType in ['badnet_sq', 'badnet_grid', 'clean']:
            return None
        
        else:
            if triggerType == 'smooth':
                trigger = imread(os.path.join('trigger', '%s_cifar10.png' % triggerType))
            else:
                trigger = imread(os.path.join('trigger', '%s.png' % triggerType)) 
            if trigger.shape[0] != 32 or trigger.shape[1] != 32:
                trigger = resize(trigger, (32,32)) # ndarray, shape=(32, 32, 3)
            trigger = img_as_ubyte(trigger)

            return trigger
