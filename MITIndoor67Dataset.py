from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
# from imgaug import augmenters as iaa

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class MITIndoor67Dataset(Dataset):
    """Class for MIT Indoor 67 dataset."""

    def __init__(self, root_dir, set, transforms_img,tencrops=False):
        
        self.image_dir = root_dir
        self.set = set
        # Set boolean variable of ten crops for validation
        self.TenCrop = tencrops

        # self.SemRGB = SemRGB
        # if SemRGB:
        #     self.RGB = "_RGB"
        # else:
        #     self.RGB = ""

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "scene_names.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]

                self.classes.append(line[split_indices[1] + 1:])

        # Get number of classes
        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(root_dir, (set + ".txt"))

        # Fill filenames list and ground-truth labels list
        with open(filenames_file) as class_file:
            for line in class_file:
                # if random.random() > 0.6 or (self.set is "val"):
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Obtain name and label
                name = line[split_indices[1] + 1:-1]
                label = line[split_indices[0] + 1: split_indices[1]]

                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)

        # ----------------------------- #
        #     ImAug Transformations     #
        # ----------------------------- #
        # Transformations for train set
        # self.seq = iaa.Sequential([
        #     # Small gaussian blur with random sigma between 0 and 0.5.
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        #     # Strengthen or weaken the contrast in each image.
        #     iaa.ContrastNormalization((0.75, 1.5)),
        #     # Add gaussian noise.
        #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        #     # Make some images brighter and some darker.
        #     iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # ], random_order=True)  # apply augmenters in random order

        # self.seq_sem = iaa.Sequential([
        #     iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        # ], random_order=True)

        # # ----------------------------- #
        # #    Pytorch Transformations    #
        # # ----------------------------- #
        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.resizeSize = 256
        self.outputSize = 224

        # Train Set Transformation
        self.train_transforms_img =transforms_img
        
        # self.train_transforms_scores = transforms.ToTensor()

        # if not SemRGB:
        #     self.train_transforms_sem = transforms.Lambda(
        #         lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
        # else:
        #     self.train_transforms_sem = transforms.Lambda(
        #         lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))

        # # Transformations for validation set
        if not self.TenCrop:
             self.val_transforms_img = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711)),
            ])

        #     if not SemRGB:
        #         self.val_transforms_sem = transforms.Compose([
        #             transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
        #             transforms.CenterCrop(self.outputSize),
        #             transforms.Lambda(lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
        #         ])

        #         self.val_transforms_scores = transforms.Compose([
        #             transforms.Resize(self.resizeSize),
        #             transforms.CenterCrop(self.outputSize),
        #             transforms.ToTensor(),
        #         ])
        #     else:
        #         self.val_transforms_sem = transforms.Compose([
        #             transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
        #             transforms.CenterCrop(self.outputSize),
        #             transforms.Lambda(lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))
        #         ])

        #         self.val_transforms_scores = transforms.Compose([
        #             transforms.Resize(self.resizeSize),
        #             transforms.CenterCrop(self.outputSize),
        #             transforms.ToTensor(),
        #         ])

        else:
            self.val_transforms_img = transforms.Compose([
                transforms.Resize(self.resizeSize),
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

        #     if not SemRGB:
        #         self.val_transforms_sem = transforms.Compose([
        #             transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
        #             transforms.TenCrop(self.outputSize),
        #             transforms.Lambda(lambda crops: torch.stack(
        #                 [torch.unsqueeze(torch.from_numpy(np.asarray(crop) + 1).long(), 0) for crop in crops]))
        #         ])

        #         self.val_transforms_scores = transforms.Compose([
        #             transforms.Resize(self.resizeSize),
        #             transforms.TenCrop(self.outputSize),
        #             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #         ])
        #     else:
        #         self.val_transforms_sem = transforms.Compose([
        #             transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
        #             transforms.TenCrop(self.outputSize),
        #             transforms.Lambda(lambda crops: torch.stack(
        #                 [torch.from_numpy(np.asarray(crop) + 1).long().permute(2, 0, 1) for crop in crops])),
        #         ])

        #         self.val_transforms_scores = transforms.Compose([
        #             transforms.Resize(self.resizeSize),
        #             transforms.TenCrop(self.outputSize),
        #             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #         ])

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Function to get a sample from the dataset. First both RGB and Semantic images are read in PIL format. Then
        transformations are applied from PIL to Numpy arrays to Tensors.

        For regular usage:
            - Images should be outputed with dimensions (3, W, H)
            - Semantic Images should be outputed with dimensions (1, W, H)

        In the case that 10-crops are used:
            - Images should be outputed with dimensions (10, 3, W, H)
            - Semantic Images should be outputed with dimensions (10, 1, W, H)

        :param idx: Index
        :return: Dictionary containing {RGB image, semantic segmentation mask, scene category index}
        """

        # Get RGB image path and load it
        img_name = os.path.join(self.image_dir, self.set, self.labels[idx], self.filenames[idx])
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")
        #img = self.train_transforms_img(img)

        # Load semantic segmentation mask
        # filename_sem = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        # sem_name = os.path.join(self.image_dir, "noisy_annotations_RGB", self.set, self.labels[idx], (filename_sem + ".png"))
        # sem = Image.open(sem_name)

        # # Load semantic segmentation scores
        # filename_scores = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        # sem_score_name = os.path.join(self.image_dir, "noisy_scores_RGB", self.set, self.labels[idx], (filename_scores + ".png"))
        # semScore = Image.open(sem_score_name)

        # # Apply transformations depending on the set (train, val)
        if self.set is "train":
        
            img = self.train_transforms_img(img)
        else:
            img = self.val_transforms_img(img)

        if not self.TenCrop:
            assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize

        else:
            assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize

      
        self.sample = {'Image': img, 'Scene Index': self.classes.index(self.labels[idx])}

        return self.sample
