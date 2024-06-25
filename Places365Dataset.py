from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
# from imgaug import augmenters as iaa


class Places365Dataset(Dataset):
    """Class for Places 365 dataset."""

    def __init__(self, root_dir, set, transforms_img, tencrops=False):

        # Extract main path and set (Train or Val).
        self.image_dir = root_dir
        self.set = set
        # Set boolean variable of ten crops for validation
        self.TenCrop = tencrops

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "categories_places365.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                split_indices = [
                    i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + \
                        '-' + line[split_indices[2]+1:]

                self.classes.append(line[split_indices[1] + 1:])

        # Get number of classes
        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        self.auxiliarnames = list()
        filenames_file = os.path.join(root_dir, (set + ".txt"))

        # Fill filenames list and ground-truth labels list
        with open(filenames_file) as class_file:
            for line in class_file:
                # if random.random() > 0.6 or (self.set is "val"):
                split_indices = [
                    i for i, letter in enumerate(line) if letter == '/']
                # Obtain name and label
                name = line[split_indices[1] + 1:-1]
                label = line[split_indices[0] + 1: split_indices[1]]

                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))
                str2 = "\n"
                indx2 = line.find(str2)
                self.auxiliarnames.append(line[0:indx2])

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)


        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.outputSize = 224
        if not self.TenCrop:
            self.val_transforms_img = transforms.Compose([
                transforms.CenterCrop(self.outputSize),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])
        else:
            self.val_transforms_img = transforms.Compose([
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        
        img_name = os.path.join(os.path.join(self.image_dir, self.auxiliarnames[idx]))
        img = Image.open(img_name)

        
        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")
            
        # img = self.train_transforms_img(img)
        if self.set is "train":
          
            img = self.train_transforms_img(img)
           
        else:
            img = self.val_transforms_img(img)
        
        if not self.TenCrop:
            assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
        else:
            assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize

           

       
        self.sample = {'Image': img,  'Scene Index': self.classes.index(self.labels[idx])}

        return self.sample
