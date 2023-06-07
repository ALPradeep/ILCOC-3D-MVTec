# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from torchvision.datasets import CIFAR10
from mvtecdataloader import mvtec3d
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
# from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch

'''
class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img
'''
'''
class OCILCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(OCILCIFAR10, self).__init__(root, train, transform, target_transform, download)

        self.prevdists = self.targets

    def set_prevdist(self, prevdists):
        self.prevdists = prevdists

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target, prevdist = self.data[index], self.targets[index], self.prevdists[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, prevdist, self.logits[index]

        return img, target, prevdist
'''

class OCILCIFAR10(mvtec3d.MVTec3DTrain):
    """
    Overrides the mvtec3d dataset to change the getitem function.
    """

    def __init__(self, class_name, img_size):
        super().__init__(class_name=class_name, img_size=img_size)
        self.class_name = class_name
        self.img_paths, self.labels = self.load_dataset(class_name)

        self.prevdists = self.labels

    def set_prevdist(self, prevdists):
        self.prevdists = prevdists

    def __getitem__(self, idx):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        # img, target, prevdist = self.data[index], self.targets[index], self.prevdists[index]

        img_path, label, prevdist = self.img_paths[idx], self.labels[idx], self.prevdists[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        '''class_label = {"bagel" : 0 ,"cable_gland" : 1, "carrot" : 2,
                   "cookie" : 3, "dowel" : 4, "foam" : 5, 
                   "peach" : 6,"potato" : 7,"rope" : 8,"tire" : 9}'''

        img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        return (img, resized_organized_pc, resized_depth_map_3channel), label, prevdist


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

    '''
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
        '''

    def get_data_loaders(self):
        """
        probably not being used
        """
        # transform = self.TRANSFORM
        class_label = {0: "bagel", 1: "cable_gland", 2: "carrot",
                       3: "cookie", 4: "dowel", 5: "foam",
                       6: "peach", 7: "potato", 8: "rope", 9: "tire"}

        class_size_dict = {0: (800, 800), 1: (400, 400), 2: (800, 800),
                           3: (500, 500), 4: (400, 400), 5: (900, 900),
                           6: (600, 600), 7: (800, 800), 8: (900, 400), 9: (600, 800)}

        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train = []
        test = []
        # for i in curr_class:
        for i in range(len(class_label)):

            train_dataset = OCILCIFAR10(class_label[i], class_size_dict[i])
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            else:
                # test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                # download=True, transform=test_transform)
                test_dataset = mvtec3d.get_data_loader("test", class_label[i], class_size_dict[i])

            train.append(train_dataset)
            test.append(test_dataset)

        # train = train[0]+train[1]
        concatenated_train = ConcatDataset(train)
        train = DataLoader(dataset=concatenated_train, batch_size=1, shuffle=False, num_workers=1,
                           drop_last=False, pin_memory=True)
        # test = test[0]+test[1]
        concatenated_test = ConcatDataset(test)
        test = DataLoader(dataset=concatenated_test, batch_size=1, shuffle=False, num_workers=1,
                          drop_last=False, pin_memory=True)
        # print(train)
        train, test = store_masked_loaders(train, test, self)

        return train, test

    """
    def get_OCIL_loaders(self): ######################
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = OCILCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        print("###################      TRAIN       #################")
        print(train)
        # train.shape()
        print("###################      TEST        #################")
        print(test)
        # test.shape()
        return train, test
        """

    def get_OCIL_loaders(self, curr_class): ######################
        # transform = self.TRANSFORM
        class_label = {0: "bagel", 1: "cable_gland", 2: "carrot",
                       3: "cookie", 4: "dowel", 5: "foam",
                       6: "peach", 7: "potato", 8: "rope", 9: "tire"}

        class_size_dict = {0: (800, 800), 1: (400, 400), 2: (800, 800),
                           3: (500, 500), 4: (400, 400), 5: (900, 900),
                           6: (600, 600), 7: (800, 800), 8: (900, 400), 9: (600, 800)}

        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train = []
        test = []
        for i in curr_class:
            # for i in range(len(class_label)):
            train_dataset = OCILCIFAR10(class_label[i], class_size_dict[i])
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            else:
                 # test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                            #download=True, transform=test_transform)
                test_dataset = mvtec3d.get_data_loader("test", class_label[i], class_size_dict[i])

            train.append(train_dataset)
            test.append(test_dataset)

        # train = train[0]+train[1]
        concatenated_train = ConcatDataset(train)
        train = DataLoader(dataset=concatenated_train, batch_size=1, shuffle=False, num_workers=1,
                           drop_last=False, pin_memory=True)
        # test = test[0]+test[1]
        concatenated_test = ConcatDataset(test)
        test = DataLoader(dataset=concatenated_test,batch_size=1, shuffle=False, num_workers=1,
                          drop_last=False, pin_memory=True)
        # print(train)
        # train, test = store_masked_loaders(train, test, self)
        #
        return train, test

    """def my_mvtec_3d(self, split, class_name, img_size):

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader"""

    """def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader"""

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
