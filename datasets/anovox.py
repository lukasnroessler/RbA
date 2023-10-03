import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


class AnoVox(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    # CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
    #                                                  'has_instances', 'ignore_in_eval', 'color'])
    # classes = [
    #     CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    #     CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    #     CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    #     CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    #     CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    #     CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    #     CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    #     CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    #     CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    #     CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    #     CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    #     CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    #     CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    #     CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    #     CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    #     CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    #     CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    #     CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    #     CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    #     CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    #     CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    #     CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    #     CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    #     CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    #     CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    #     CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    #     CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    #     CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    #     CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    #     CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    # ]

    # train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    # train_id_to_color.append([0, 0, 0])
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.train_id for c in classes])
    
    def __init__(self, hparams, split='train', transform=None):
        
        self.hparams = hparams
        # self.root = os.path.expanduser(hparams.dataset_root)
        self.root = hparams.dataset_root
        # self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        
        # self.targets_dir = os.path.join(self.root, 'gtFine', split)
        

        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        for scenario in os.listdir(self.root):
            if scenario == 'Scenario_Configuration_Files':
                continue
            image_dir = os.path.join(self.root, scenario, 'RGB_IMG')
            target_dir = os.path.join(self.root, scenario, 'SEMANTIC_IMG')
            files_name = sorted(os.listdir(image_dir))
            for image_file_name in files_name:
                self.images.append(os.path.join(image_dir, image_file_name))

                target_file_name = image_file_name.replace('RGB', 'SEMANTIC')
                self.targets.append(os.path.join(target_dir, target_file_name))





        # if split not in ['train', 'val', 'test']:
        #     raise ValueError('Invalid split for mode! Please use split="train", split="test"'
        #                      ' or split="val"')

        # if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
        #     raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
        #                        ' specified "split" and "mode" are inside the "root" directory')

        
        # for scenario in os.listdir(self.images_dir):
        #     img_dir = os.path.join(self.images_dir, scenario)
        #     target_dir = os.path.join(self.targets_dir, scenario)
        #     files_name = os.listdir(img_dir)
        #     files_name = sorted(files_name)
        #     for file_name in files_name:
        #         self.images.append(os.path.join(img_dir, file_name))
        #         target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
        #                                      'gtFine_labelTrainIds.png')
        #         self.targets.append(os.path.join(target_dir, target_name))


    # @classmethod
    # def encode_target(cls, target):

    #     target = cls.id_to_train_id[np.array(target)]
    #     target[target == 255] = 19
    #     return target

    # @classmethod
    # def decode_target(cls, target):
    #     #target[target == 255] = 19
    #     #target = target.astype('uint8') + 1
    #     return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        # anomaly_color = [245,0,0] # official anomaly color
        # anomaly_color = (170, 120, 50) # brown, dynamic
        # anomaly_color = (110, 190, 160) # turqoise, static
        anomaly_color = [184,15,10] # special

        # image = np.array(Image.open(self.images[index]).convert('RGB'))
        image = np.array(Image.open(self.images[index]).convert('RGB'))
        target = np.array(Image.open(self.targets[index]))
        label = np.zeros((target.shape), dtype=np.uint8)[:,:,:1]
        label = np.squeeze(label)
        anomaly_mask = (target[:, :,None] == anomaly_color).all(-1).any(-1)
        label[anomaly_mask] = 1

        # target[target == 255] = 19

        if self.transform is not None:
            aug = self.transform(image=image, mask=label)
            image, label = aug["image"], aug["mask"]
       
        return image, label.type(torch.LongTensor)
    
    def __len__(self):
        return len(self.images)