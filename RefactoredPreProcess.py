import os
import cv2
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import convolve
from abc import ABC, abstractmethod

class ImageLoader(ABC):

    @abstractmethod
    def Load(self):
        pass
class DicomLoader(ImageLoader):

    def Load(self):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.folder)
        image = reader.Execute()

        return (image,1)

class VLoader(ImageLoader):

    def Load(self):
        pass

class ILoader(ImageLoader):

    def Load(self):
        pass
class NiftiLoader(ImageLoader):

    def Load(self):
        pass
        
class ImageLoaderFactory:

    _loaders = {
        ".dicom": DicomLoader,
        ".v": VLoader,
        ".i": ILoader,
        ".nii": NiftiLoader
    }

    @classmethod
    def create(cls, fmt, folder):

        try:
            loader_class = cls._loaders[fmt]
        except KeyError:
            raise ValueError(f"Unknown format: {fmt}")

        return loader_class(folder)
class DatasetLoader:

    def __init__(self, mri_dict, pet_dict):
        self.mri_dict = mri_dict
        self.pet_dict = pet_dict

    def LoadDataset(self, number=None):

        dataset = []

        items = list(
            zip(
                self.mri_dict.items(),
                self.pet_dict.items()
            )
        )

        if number is not None:
            items = items[:number]

        print(len(items))

        pair_counter = defaultdict(int)
        Info = []

        for (mri_folder, mri_fmt), (pet_folder, pet_fmt) in items:

            mri_loader = ImageLoaderFactory.create(
                mri_folder,
                mri_fmt,
                "mri"
            )

            pet_loader = ImageLoaderFactory.create(
                pet_folder,
                pet_fmt,
                "pet"
            )

            mri, mri_frames = mri_loader.load()
            pet, pet_frames = pet_loader.load()

            dataset.append((mri, pet))

            Info.append(
                self._FindSubjectAndPairs(
                    pair_counter,
                    mri_folder
                )
            )

        return dataset, Info