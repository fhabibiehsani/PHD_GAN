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
np.set_printoptions(threshold=np.inf)
class ImageInfo():
    def __init__(self,image,type=""):
        self.image=image
        self.type=type
    def ImageInfo(self):
        self._GetOrientartion()
        self._GetOrigin()
        self._GetSpacing()
        self._GetDirection()  
        self._GetDicomImageSize()   
    def _GetOrientartion(self):
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(self.image.GetDirection())
        print("Detected "+self.type+" Image Orientartion:", orientation)                                                                
    def _GetOrigin(self):
        origin= self.image.GetOrigin()
        print("Detected "+self.type+" Image Origin:", origin)
    def _GetSpacing(self):
        spacing=self.image.GetSpacing()
        print("Detected "+self.type+" Image Spacing:", spacing)
    def _GetDirection(self):
        direction=self.image.GetDirection()
        print("Detected "+self.type+" Image Direction:",direction)
    def _GetDicomImageSize(self):
        image = sitk.DICOMOrient(self.image, "LPS")
        arr = self.ConvertToNumpy(image)

        # number of slices
        z = arr.shape[0]
        y = arr.shape[1]
        x = arr.shape[2]

        print("Shape (z, y, x):", arr.shape)
        print("Number of slices (z):", z)

        return image, arr     
class ImageProcessor():
    def __init__(self,image):
        self.image=image
    def NormalizeTanh(self):
        image_min = self.image.min()
        image_max = self.image.max()
        self.image = (self.image - image_min) / (image_max - image_min )  # [0,1]
        normalizedImage = self.image * 2.0 - 1.0                               # [-1,1]
        return normalizedImage
    def MedianFilter(self,size=5):
        filteredImage = cv2.medianBlur(self.image, size)  # kernel size must be odd: 3,5,7,...
        return filteredImage
    def GaussianFilter(self):
        image = sitk.Median(self.image, [3, 3, 3]) 
        gaussianFilteredImage = sitk.DiscreteGaussian(image, variance=1.0)
        return gaussianFilteredImage
    def BiasFieldCorrectionN4(self):
        image = sitk.GetImageFromArray(self.image)
        # Convert to float (IMPORTANT)
        image = sitk.Cast(image, sitk.sitkFloat32)
        # Initial mask (foreground estimation)
        maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
        # N4 Bias Field Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        correctedImage = corrector.Execute(image, maskImage)
        # Convert back to numpy
        biasFieldCorrectionN4Image = sitk.GetArrayFromImage(correctedImage)

        return biasFieldCorrectionN4Image
    def ConvertToNumpy(self):

        if self.image is None:
            raise ValueError("Image is None")

        if isinstance(self.image, np.ndarray):
            return self.image

        if isinstance(self.image, sitk.Image):
            return sitk.GetArrayFromImage(self.image)

        raise TypeError(f"Unsupported type: {type(self.image)}")
    def HistogramMatch(self, image, reference):
        matcher = sitk.HistogramMatchingImageFilter()

        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(50)
        matcher.ThresholdAtMeanIntensityOn()

        matched = matcher.Execute(
            image,
            reference
        )

        return matched
class ImageLoader():
    def __init__(self,folder ,fmt, mode):
        self.folder=folder
        self.fmt=fmt
        self.mode=mode
    def LoadImages(self):
        if self.fmt == ".dicom":
            return self.Load_dicom_file()
        elif self.fmt == ".v":
            return self.Load_v_file()
        elif self.fmt == ".i":
            return self.Load_i_file()
        elif self.fmt == ".gz":
            return self.Load_NIfTI_Image_File()
        elif self.fmt == ".nii":
            return self.Load_NIfTI_Image_File()
        else:
            raise ValueError("Unknown format")
    def Load_dicom_file(self, frames=1):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.folder)
        n = len(dicom_names)
        if n == 0:
            return None

        if self.mode=="pet" and self.IsDynamicPETImage():
            dicom_names = dicom_names[:n // 6]
            frames=6
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except:
            return None
        
        image = sitk.DICOMOrient(image, "RAI")      #Orientation : RAS   #LPS
        image=self.ConvertImageToFloat32(image)     #Type conversion
        image = sitk.RescaleIntensity(image, 0, 1)
        image = sitk.DiscreteGaussian(image, variance=1.0)
        original = image
        arr = sitk.GetArrayFromImage(image)             
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) 
        arr = np.where(arr < 0.01, 0, arr)
        image = sitk.GetImageFromArray(arr)
        image.CopyInformation(original)
        if self.mode=="pet" and (image.GetSize()==(128, 128, 47) or image.GetSize()==(400, 400, 109) or image.GetSize()==(128, 128, 31)):
            image = sitk.DICOMOrient(image, "LAS") 
            image.SetDirection([-1,0,0, 0,-1,0, 0,0,-1]) 
        return (image,frames)
    def Load_v_file(self, frames=6, slices=63, rows=128, cols=128, header=0):
        v_files = []
        for root, _, files in os.walk(self.folder):
            for f in files:
                if f.endswith(".v"):
                    v_files.append(os.path.join(root, f))

        if len(v_files) == 0:
            raise FileNotFoundError("No .v file found")

        # read data
        file_path = v_files[0]
        data = np.fromfile(file_path, dtype='>i2', offset=header)
        data = data.astype(np.float32)
    
        #print("Raw Data size:", data.size)
        

        base = slices * rows * cols
        frames = data.size // base
        usable_size = frames * base
        #if (data.size-usable_size!=2048):
        #    print( data.size-usable_size)
        #    print("Raw Data size:", data.size)
        #    print("Detected frames:", frames)

        if frames == 0:
            raise ValueError("❌ Cannot reshape data")

        #trim extra data
        #data = data[-usable_size:]
        data = data[-usable_size:]
        #print("Original size:", data.size)
        #print("Usable size:", usable_size)
        #print("Removed extra:", data.size - usable_size)

        # reshape safely
        vol = data.reshape((frames, slices, rows, cols))

        # take first frame
        arr = vol[-1]
        arr = np.flip(arr, axis=1)  #This flip images
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) 
        arr = np.where(arr < 0.01, 0, arr)
        
        # convert to SimpleITK
        image = sitk.GetImageFromArray(arr)
        image = sitk.RescaleIntensity(image, 0, 1)
        image = sitk.DiscreteGaussian(image, variance=1.0)
        image = sitk.DICOMOrient(image, "RPS")  
        # todo
        if ("30_min_3D_FDG" in v_files[0] or "Dynamic_2_FDG" in v_files[0]  or  "30MIN_3D_FDG" in v_files[0] or  "ADNI_FDG_ITER_img" in v_files[0]): #
            # print(v_files)
            image.SetSpacing((2.1, 2.1, 2.4))
        else:
            image.SetSpacing((2.6, 2.6, 2.4))
        image.SetOrigin((128.0, 128.0, 75.6))
        image.SetDirection([-1,0,0, 0,-1,0,0,0,-1])
        return (image,frames)
    def Load_i_file(self, frames=1 ,slices=207, rows=256,cols=256,header=0):
        i_files = []
        for root, _, files in os.walk(self.folder):
            for f in files:
                if f.endswith(".i"):
                    i_files.append(os.path.join(root, f))

        if len(i_files) == 0:
            raise FileNotFoundError("No .i file found")
        
        # read data
        file_path = i_files[0]
        data = np.fromfile(file_path, dtype=np.float32, offset=header)
        data = data.astype(np.float32)
        #print("Raw Data size:", data.size)

        base = slices * rows * cols

        frames = data.size // base
        usable_size = frames * base

        #print("Detected frames:", frames)

        if frames == 0:
            raise ValueError("❌ Cannot reshape data")

        data = data[-usable_size:]

        # reshape safely
        vol = data.reshape((frames, slices, rows, cols))

        # take first frame
        arr = vol[-1]
        arr = np.flip(arr, axis=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) 
        arr = np.where(arr < 0.01, 0, arr)  #all pix less than 0.01 set 0
        # convert to SimpleITK
        image = sitk.GetImageFromArray(arr)
        image = sitk.RescaleIntensity(image, 0, 1)
        image = sitk.DiscreteGaussian(image, variance=1.0)
        image = sitk.DICOMOrient(image, "RPS")
        image.SetSpacing((1.2, 1.2,1.2))
        image.SetOrigin((128.0, 128.0, 75.6))
        image.SetDirection([-1,0,0,0,-1,0,0,0,-1])
        return (image,frames)
    def Load_NIfTI_Image_File(self):
        image = sitk.ReadImage(self.folder)
        if image.GetDimension() == 3:
            image = sitk.DICOMOrient(image, "RAI")
            image.SetDirection([-1,0,0, 0,-1,0, 0,0,-1])
        else:
            image.SetDirection([-1,0, 0,-1])
        image = self.ConvertImageToFloat32(image)
        image = sitk.RescaleIntensity(image, 0, 1)
        return (image, 1)
    def IsDynamicPETImage(self):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.folder)

        reader.SetFileNames(dicom_names)

        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        reader.Execute()

        try:
            num_frames = int(reader.GetMetaData(0, "0054|0101"))  # Number of Time Slices
            return num_frames > 1
        except:
            return False
    def ConvertImageToFloat32(self,image):
        image = sitk.Cast(image, sitk.sitkFloat32)
        return image          
class DatasetExplorer():

    def __init__(self,root_folder,dim):
        super().__init__()
        self.root_folder = root_folder
        self.dim=dim
        if self.dim=="3D":
            self.mri_root=os.path.join(self.root_folder, "MRI214", "ADNI")
            self.pet_root=os.path.join(self.root_folder, "PET214", "ADNI")
            self.subject=self.pet_root
        elif self.dim=="2D":
            self.mri_root=os.path.join(self.root_folder, "mri.nii.gz")
            self.pet_root=os.path.join(self.root_folder, "pet.nii.gz")
            self.subject= self.root_folder
        elif self.dim=="2D_skip":
            self.mri_root=self.root_folder
            self.pet_root=self.root_folder
            self.subject= self.root_folder

    def GetSubjects(self):
        subjects= os.listdir(self.subject)
        print("Total subjects:", len(subjects))
        return subjects
    def FindPairFolders(self,file_format=None):
        subjects = self.GetSubjects()

        mri_dict = {}
        pet_dict = {}

        for subject in subjects:
            if self.dim=="3D":
                mri_path = os.path.join(self.mri_root, subject)
                pet_path = os.path.join(self.pet_root, subject)
                if not os.path.isdir(mri_path) or not os.path.exists(pet_path):
                    continue

                # ---------- MRI ----------
                for root, _, _ in os.walk(mri_path):
                    fmt = self.DetectFileExtension(root)
                    if fmt and (file_format is None or fmt == file_format):
                        mri_dict[root] = fmt

                # ---------- PET ----------
                for root, _, _ in os.walk(pet_path):
                    fmt = self.DetectFileExtension(root)
                    if fmt and (file_format is None or fmt == file_format):
                        pet_dict[root] = fmt
            elif self.dim=="2D":
                for subject in os.listdir(self.root_folder):

                    subject_path = os.path.join(self.root_folder, subject)

                    if not os.path.isdir(subject_path):
                        continue

                    for pair in os.listdir(subject_path):

                        pair_path = os.path.join(subject_path, pair)

                        mri_path = os.path.join(pair_path, "mri.nii.gz")
                        pet_path = os.path.join(pair_path, "pet.nii.gz")
                        # ✔ check files exist
                        # if not os.path.exists(mri_path):
                        #     print("Missing MRI:", mri_path)
                        #     continue

                        # if not os.path.exists(pet_path):
                        #     print("Missing PET:", pet_path)
                        #     continue
                        fmt = self.DetectFileExtension(pet_path)
                        if fmt and (file_format is None or fmt == file_format):
                            pet_dict[pet_path] = fmt
                        fmt = self.DetectFileExtension(mri_path)
                        if fmt and (file_format is None or fmt == file_format):
                            mri_dict[mri_path] = fmt
            elif self.dim=="2D_skip":
                for subject in os.listdir(self.root_folder):
                        mri_path = os.path.join(self.root_folder, subject,"mri.nii")
                        pet_path = os.path.join(self.root_folder,subject,"pet.nii")
                        
                        fmt = self.DetectFileExtension(pet_path)
                        if fmt and (file_format is None or fmt == file_format):
                            pet_dict[pet_path] = fmt
                        fmt = self.DetectFileExtension(mri_path)
                        if fmt and (file_format is None or fmt == file_format):
                            mri_dict[mri_path] = fmt
        return mri_dict,pet_dict
  
    def FindAllFileTypes(self):
        extensions_Pet=self._FindAllTypes(self.pet_root,"PET")
        self._CountDifferentFileTypes(self.pet_root,"PET")
        extensions_mri=self._FindAllTypes(self.mri_root,"MRI")
        self._CountDifferentFileTypes(self.mri_root,"MRI")
    def _FindAllTypes(self,path,name):
        extensions = {}

        for root, _, files in os.walk(path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()

                if ext == "":
                    ext = "NO_EXTENSION"

                extensions[ext] = extensions.get(ext, 0) + 1

        print(f"{name} File types:")
        for k, v in sorted(extensions.items(), key=lambda x: -x[1]):
            print(k, ":", v)
        print("--------------------")
        return extensions
    def _CountDifferentFileTypes(self,path,name):
        dicom_folders = set()
        v_folders = set()
        i_folders = set()
        nii_folders=set()
        for root, _, files in os.walk(path):

            has_dicom = False
            has_v = False
            has_i = False
            has_nii = False
            for f in files:
                f_lower = f.lower()

                # DICOM (file-based or extension-based)
                if f_lower.endswith(".dcm") or f_lower.endswith(".dicom"):
                    has_dicom = True

                # V files
                if f_lower.endswith(".v"):
                    has_v = True
                if f_lower.endswith(".i"):
                    has_i = True
                if f_lower.endswith(".nii"):
                    has_nii = True

            # One time for each Folder
            if has_dicom:
                dicom_folders.add(root)
            if has_v:
                v_folders.add(root)
            if has_i:
                i_folders.add(root)
            if has_nii:
                nii_folders.add(root)

        print(f"{name} Folders Counts:")
        print("DICOM folders:", len(dicom_folders))
        print("V folders:", len(v_folders))
        print("i folders:", len(i_folders))
        print("nii folders:", len(nii_folders))
        print("Total medical folders:", len(dicom_folders) + len(v_folders)+ len(i_folders)+ len(nii_folders))
        print("--------------------")

    def CheckAnyUniqueImageSize(self, dtype='>i2'):
        size_counts = {}   
        examples = {}     

        for root, _, files in os.walk(self.pet_root):
            for f in files:
                if f.lower().endswith(".i"):
                    file_path = os.path.join(root, f)

                    try:
                        file_size = os.path.getsize(file_path)

                        best_size = None

                        # Try dif Header
                        for header in [0, 1024, 2048, 4096, 8192]:
                            data = np.fromfile(file_path, dtype=dtype, offset=header)
                            size = data.size
                            print(size)

                            if best_size is None:
                                best_size = size

                        #Count
                        if best_size not in size_counts:
                            size_counts[best_size] = 0
                            examples[best_size] = file_path

                        size_counts[best_size] += 1

                    except Exception as e:
                        print("❌ Error:", file_path, e)
        print(best_size)
        print("\nUnique sizes with counts:\n")

        for size, count in sorted(size_counts.items()):
            print(f"Size: {size}  →  Count: {count}")
            print(f"Example file: {examples[size]}")
            print("-" * 50)

        print("Total unique sizes:", len(size_counts))
    def DetectFileExtension(self,path):
        # If path is a file
        if os.path.isfile(path):

            if path.lower().endswith(".nii"):
                return ".nii"

            return os.path.splitext(path)[1].lower()

        # If path is a directory
        elif os.path.isdir(path):

            files = os.listdir(path)

            if self.IsFileExtension('.dcm', files):
                return ".dicom"
            elif self.IsFileExtension('.v', files):
                return ".v"
            elif self.IsFileExtension('.i', files):
                return ".i"
            elif self.IsFileExtension('.hdr', files):
                return ".hdr"
            elif self.IsFileExtension('.nii', files):
                return ".nii"

        return None
    def IsFileExtension(self,extension,files):
        return any(f.lower().endswith(extension) for f in files)
    def HasTimeDimention(self):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.root_folder)

        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        reader.SetFileNames(dicom_names)

        reader.Execute()

        times = []

        for i in range(len(dicom_names)):
            if reader.HasMetaDataKey(i, "0054|1300"):  # Frame Reference Time
                times.append(reader.GetMetaData(i, "0054|1300"))

        return len(set(times)) > 1
    def DetectPETImageType(self):
        if self.IsDynamicPETImage(self.root_folder):
            return "Dynamic PET"
        elif self.HasTimeDimention(self.root_folder):
            return "Dynamic PET"
        else:
            return "Static PET"
class DatasetLoader():
    def __init__(self,mri_dict, pet_dict):
        super().__init__()
        self.mri_dict = mri_dict
        self.pet_dict = pet_dict
    def LoadDataset(self,number=None):
      
        dataset = []
        items = list(zip(self.mri_dict.items(), self.pet_dict.items()))
        if number is not None:
            items = items[:number]
        print(len(items))
        pair_counter = defaultdict(int)
        Info = []
        for (mri_folder, mri_fmt), (pet_folder, pet_fmt) in items:
            if pet_fmt==".dicom":

                mriImageLoader=ImageLoader(mri_folder, mri_fmt, "mri")
                petImageLoader=ImageLoader(pet_folder, pet_fmt, "pet")

                mri, mri_frames = mriImageLoader.LoadImages()
                pet, pet_frames = petImageLoader.LoadImages()

                dataset.append((mri, pet))
                Info.append(self._FindSubjectAndPairs(pair_counter,mri_folder))
        return (dataset,Info)
    def _FindSubjectAndPairs(self,pair_counter,mri_folder):
        parts = Path(mri_folder).parts
        adni_index = parts.index("ADNI")

        subject = parts[adni_index + 1]
        pair = parts[adni_index + 2]

        # Start from 1 for each subject
        pair_counter[subject] += 1
        pair_number = pair_counter[subject]

        # print(subject, pair_number)
        return (subject, pair_number)
        # return ("",1)
 
    def Load_nii_Dataset(self,input_nii):

        for subjectCounter,subject in  enumerate(os.listdir(input_nii)):

            subject_path = os.path.join(input_nii, subject)

            if not os.path.isdir(subject_path):
                continue

            for pairCounter, pair in enumerate(os.listdir(subject_path)):

                pair_path = os.path.join(subject_path, pair)

                mri_path = os.path.join(pair_path, "mri.nii.gz")
                pet_path = os.path.join(pair_path, "pet.nii.gz")

                # ✔ check files exist
                if not os.path.exists(mri_path):
                    print("Missing MRI:", mri_path)
                    continue

                if not os.path.exists(pet_path):
                    print("Missing PET:", pet_path)
                    continue


                #-------------------------------------

                # ✔ read images 
                mri = self.Load_NIfTI_Image_File(mri_path)
                pet = self.Load_NIfTI_Image_File(pet_path)
                mri_np=self.convert_to_NumPy(mri)
                pet_np=self.convert_to_NumPy(pet)
                print("MRI min:", mri_np.min())
                print("MRI max:", mri_np.max())
                print("PET min:", pet_np.min())
                print("PET max:", pet_np.max())
                title=f"Subject({subjectCounter+1}):{subject}, Pair({pairCounter+1})"
                #plot_pair_2D(mri, pet,title)
                mri=self.minmax_normalize(mri)
                pet=self.minmax_normalize(pet)
                self.Hist_Pair_2D(mri*mri, pet*pet)
                mri_masked, pet_masked=self.Max_bin_Set_One(mri, pet)
                self.Hist_Pair_2D(mri_masked, pet_masked)
                self.plot_pair_2D(mri_masked, pet_masked,title)
                print("mri_masked min:", mri_masked.min())
                print("mri_masked max:", mri_masked.max())
                print("pet_masked min:", pet_masked.min())
                print("pet_masked max:", pet_masked.max())
                #break
        # break
    def Load_nii_files_fromDataset(self,output_2D_nii_Dataset):
        for subject in os.listdir(output_2D_nii_Dataset):

            subject_path = os.path.join(output_2D_nii_Dataset, subject)

            if not os.path.isdir(subject_path):
                continue

            for pair in os.listdir(subject_path):

                pair_path = os.path.join(subject_path, pair)

                mri_path = os.path.join(pair_path, "mri.nii.gz")
                pet_path = os.path.join(pair_path, "pet.nii.gz")

                # ✔ check files exist
                if not os.path.exists(mri_path):
                    print("Missing MRI:", mri_path)
                    continue

                if not os.path.exists(pet_path):
                    print("Missing PET:", pet_path)
                    continue

                # ✔ read images
                mri = sitk.ReadImage(mri_path)
                pet = sitk.ReadImage(pet_path)
                # To Do
                print("Loaded:", subject, pair)  
class Preprocessor():
    def __init__(self,dataset,Info,index=None):
        self.dataset = dataset if index is None else dataset[:index]
        self.Info=Info
        #3D Dataset
    def Extract2Ddatasetfrom3DdatasetperSliceIds(self,sliceIds):
            new2Ddataset=[]
            for sliceId in sliceIds:
                dataset=self._Extract2Ddatasetfrom3Ddataset(sliceId)
                new2Ddataset.extend(dataset)
            return new2Ddataset
    def _Extract2Ddatasetfrom3Ddataset(self,sliceId=None):
        new2Ddataset = []
        for mri_img, pet_img in self.dataset:
            mri=Image3D(mri_img,"MRI")
            pet=Image3D(pet_img,"PET")

            if sliceId is None:
                mri2D,mriSliceId=mri.GetMidSlice()
                pet2D,petSliceId=pet.GetMidSlice()
            else:
                # for sliceId in sliceIds:
                    mri2D,mriSliceId=mri.GetSliceId(sliceId)
                    pet2D,petSliceId=pet.GetSliceId(sliceId)
            mri2D = sitk.GetImageFromArray(mri2D)
            pet2D = sitk.GetImageFromArray(pet2D)
            new2Ddataset.append((mri2D, pet2D))   

        return new2Ddataset

    def RegisterDataset(self,MNI):
        registeredDataset = []
        counter=1
        for mri, pet in self.dataset:
            print(counter)
            counter+=1
            registeredmri= self._RegisterPetToMri(MNI, mri)
            registeredPet= self._RegisterPetToMri(registeredmri, pet)
            registeredDataset.append((registeredmri, registeredPet))

        self.dataset = registeredDataset
    def _RegisterPetToMri(self,MRI,PET):
        
        # ---------- initial alignment ----------
        initial_transform = sitk.CenteredTransformInitializer(
            MRI,   # fixed (target space)
            PET,   # moving
            sitk.Euler3DTransform(),                           #rotation چرخش and translation انتقال
            sitk.CenteredTransformInitializerFilter.GEOMETRY   #.GEOMETRY مرکز تصویر را بر اساس هندسه (ابعاد و سایز) تنظیم می‌کند  
            # به شدت پیکسل وابسته نیست (برخلاف MOMENTS) MOMENTS
        )
        #print(sitk.CenteredTransformInitializer(mri, pet, sitk.Euler3DTransform()))

        # ---------- registration method ----------
        registration = sitk.ImageRegistrationMethod()

        ## similarity metric (خوب برای multi-modality)
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # optimizer      
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.5,
            minStep=1e-4,
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-6
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        # interpolation
        registration.SetInterpolator(sitk.sitkLinear)

        # initial transform
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # multiresolution     
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # ---------- run registration ----------
        final_transform = registration.Execute(MRI,  PET)
        identity = sitk.Transform(3, sitk.sitkIdentity)
        # ---------- resample PET into MRI space ----------
        pet_registered = sitk.Resample(
        PET,
        MRI,         # fixed reference
        final_transform,
        sitk.sitkLinear,
        0.0,
        PET.GetPixelID()
    )
        return pet_registered

    def PlotPairs3DViews(self):
        print(len(self.dataset))
        for i, ((mri, pet), (SubjectId, pairNumber)) in enumerate(zip(self.dataset, self.Info)):
            mri=Image3D(mri,"MRI")
            pet=Image3D(pet,"PET")
            mri.Plot3DViews(i+1,SubjectId,pairNumber)
            pet.Plot3DViews(i+1,SubjectId,pairNumber)
    
    def PlotPairsMidSlice(self):
        for i, ((mri, pet), (SubjectId, pairNumber)) in enumerate(zip(self.dataset, self.Info)):
            mri=Image3D(mri,"MRI")
            pet=Image3D(pet,"PET")
            mri_s, mri_idx = mri.GetMidSlice()
            pet_s, pet_idx = pet.GetMidSlice()
            pair=PairImage2D(mri_s,pet_s,i+1)
            pair.Plot2DPairs(i+1,SubjectId,pairNumber)
    def PlotPairsSliceID(self,SliceId):
         for i, ((mri, pet), (SubjectId, pairNumber)) in enumerate(zip(self.dataset, self.Info)):
            mri=Image3D(mri)
            pet=Image3D(pet)
            mri_s, mri_idx = mri.GetSliceId(SliceId)
            pet_s, pet_idx = pet.GetSliceId(SliceId)
            pair=PairImage2D(mri_s,pet_s,i+1)
            pair.Plot2DPairs(i+1,SubjectId,pairNumber)

    def skull_strip_remove_neck(input_img, output_img, frac=0.3):
            cmd = [
                "bet",
                input_img,
                output_img,
                "-R",   # robust center estimation
                "-B",   # bias field + better cleanup
                "-f", str(frac),  # fractional intensity (try 0.25–0.4)
                "-g", "0",         # vertical gradient (important for neck removal)
                "-m"   # also output mask
            ]
            subprocess.run(cmd, check=True)
    def GetDataset(self):
        return self.dataset
class Visualizer():
    def __init__(self,dataset,index=None):
        self.dataset=dataset   
        self.data = self.dataset if index is None else self.dataset[:index]
        self.reference_mri = self.data[0][0]
        self.reference_pet = self.data[0][1]
    def do(self):
        for i,(mri_img, pet_img) in enumerate(self.data):
            # pet_img=pet_img*pet_img
            # mri_processor = ImageProcessor(mri_img)
            # pet_processor = ImageProcessor(pet_img)
            # mri_img = mri_processor.HistogramMatch( mri_img,self.reference_mri)
            # pet_img = pet_processor.HistogramMatch( pet_img,self.reference_pet)
            
            p=PairImage2D(mri_img, pet_img,i)
            # p.GaussianNormPairs()
            # p.MaxBinSetZero()
            # p.PlotHist2()
            # p.MinMaxNormPairs()
            # 
            p.GetMinMax()
            p.Plot2DPairs()
            # p.PlotHist2()
            # p.Plot2DPairs()
              # Update dataset
            self.data[i] = ( p.mriNumpy, p.petNumpy )
    def GetPairsSize(self):
            pair=PairImage2D(self.data[0][0],self.data[0][1])
            pair.GetPairsSize()

    def Skip(self):
        skip_indices = {2,3,6,7,
                        12,15,16,
                        23,
                        34,44,51,52,53,54,55,56,58,59,
                        60,62,66,67,68,
                        70,71,72,73,78,
                        96,100,102,107,112,
                        123,129,130,131,132,133,134,135,136,138,
                        139,140,141,142,143,144,148,
                        159,181,192,201,202,203,204,206,207,208,209,
                        211,212
        }

        print(len(skip_indices))
                        # 36,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67, 69, 
                        # 70, 71, 72, 82, 83,84, 91, 104, 106, 111, 112, 113, 114, 118,
                        # 120, 122, 123, 125, 126, 128, 129, 130,138,139,140,141,142,143, 147, 149, 153, 155, 156, 157,
                        # 160, 171, 172, 177, 178, 180, 184, 188, 189, 192, 193, 194, 195,197,
                        # 206, 213} 
        self.data = [
            item for i, item in enumerate(self.data)
            if i not in skip_indices
        ]
        print(len(self.data))
    def GetDataset(self):
        return self.data
    def PlotPairImageAndHist(self):
        for i, (mri_img, pet_img) in enumerate(self.data):

            mri = Image2D(mri_img,"MRI")
            pet = Image2D(pet_img,"PET")

            mri_flat = mri.ConvertToNumpy().flatten()
            pet_flat = pet.ConvertToNumpy().flatten()
             # MRI histogram
            counts_mri, bins_mri = np.histogram(mri_flat, bins=100)
            max_idx_mri = np.argmax(counts_mri)
            counts_mri[max_idx_mri] = 0   # skip max bin

            # PET histogram
            counts_pet, bins_pet = np.histogram(pet_flat, bins=100)
            max_idx_pet = np.argmax(counts_pet)
            counts_pet[max_idx_pet] = 0   # skip max bin
            plt.figure(figsize=(20, 4))

            # 1) MRI image
            plt.subplot(1, 4, 1)
            plt.imshow(mri, cmap="gray")
            plt.title("MRI Image")
            plt.axis("off")

            # 2) PET image
            plt.subplot(1, 4, 2)
            plt.imshow(pet, cmap="gray")
            plt.title("PET Image")
            plt.axis("off")

            # 3) MRI histogram
            plt.subplot(1, 4, 3)
            plt.bar(bins_mri[:-1], counts_mri, width=np.diff(bins_mri))
            plt.title("MRI Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Count")

            # 4) PET histogram
            plt.subplot(1, 4, 4)
            plt.bar(bins_pet[:-1], counts_pet, width=np.diff(bins_pet))
            plt.title("PET Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Count")

            plt.suptitle(f"Subject {i+1}")
            plt.tight_layout()
            plt.show()
class DatasetBuilder():
    def __init__(self,dataset,output=""):
        self.output = output   
        self.dataset=dataset
    def SaveDatasetAsNII(self): 
        for subjectId, (mri, pet) in enumerate(self.dataset):

            subject_folder = os.path.join(self.output, str(subjectId+1))
            os.makedirs(subject_folder, exist_ok=True)
                # Convert NumPy → SimpleITK
            if isinstance(mri, np.ndarray):
                mri = sitk.GetImageFromArray(
                    mri.astype(np.float32)
                )

            if isinstance(pet, np.ndarray):
                pet = sitk.GetImageFromArray(
                    pet.astype(np.float32)
                )

            sitk.WriteImage(mri, os.path.join(subject_folder, "mri.nii"))
            sitk.WriteImage(pet, os.path.join(subject_folder, "pet.nii"))

        print("Saving NIfTI 3D dataset finalized.")
    def Save2DdatasetAsPNGimage(self): 
        Image_folder = os.path.join(self.output, "Image")
        os.makedirs(Image_folder, exist_ok=True)
        for subjectId, (mri, pet) in enumerate(self.dataset):
            sitk.WriteImage(mri, f"mri_{subjectId+1}.png")
            sitk.WriteImage(pet, f"pet_{subjectId+1}.png")

        print("Saving PNG 2D dataset finalized.")
    def Save3DViewsOf3DdatasetAsPNGimage(self):
        print(len(self.dataset))
        for subjectId, (mri, pet) in enumerate( self.dataset):
            mri=Image3D(mri,"MRI")
            pet=Image3D(pet,"PET")
            # mri.Save3DViews(i+1)
            # pet.Save3DViews(i+1)
class Image3D(ImageProcessor):
    def __init__(self,image,name):
        super().__init__(self)
        self.image=image
        self.name=name
        self.numpyImage=self.ConvertToNumpy()
    def Plot3DViews(self,subject="",subjectId="",pair="",path="",save=False):
       
        z, y, x = self.numpyImage.shape
        # print(self.numpyImage.shape)
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(self.numpyImage[z//2], cmap='gray')
        plt.title("Axial")

        plt.subplot(1,3,2)
        plt.imshow(self.numpyImage[:, y//2, :], cmap='gray')
        plt.title("Coronal")

        plt.subplot(1,3,3)
        plt.imshow(self.numpyImage[:, :, x//2], cmap='gray')
        plt.title("Sagittal")

        plt.suptitle(f"Subject({subject}):{subjectId}, Pair({pair})")  
        save_path = os.path.join(path,rf"architecture_batch_comparison.png")
        if save:
            plt.savefig(save_path,dpi=300,bbox_inches="tight")
        plt.show()

    def GetMidSlice(self):
        midIDNumber = self.numpyImage.shape[0] // 2
        slice = self.numpyImage[midIDNumber]
        return slice, midIDNumber   
    def GetSliceId(self,sliceId):
        if sliceId < 0 or sliceId >= self.numpyImage.shape[0]:
            raise ValueError("sliceId out of range")
        return self.numpyImage[sliceId],sliceId
    def PlotMiddleSlice(self,subject="",subjectId="",pair=""):
        print("ITK size:", self.image.GetSize())

        arr = self.ConvertToNumpy(self.image)
        arr = np.transpose(arr, (1, 2, 0))
        mid_slice = arr[arr.shape[0] // 2]
        
        print(arr.shape)
        plt.imshow(mid_slice, cmap="gray")
        plt.title(f"Subject({subject}):{subjectId}, Pair({pair})")
        plt.axis("off")
        plt.show()
    def ConvertToNumpy(self):
        return super().ConvertToNumpy()          
class MNI(Image3D):
    def __init__(self,MNI_Root):
        self.MNI_Root=MNI_Root
        self.MNI_Template=self.LoadMNIimage()
        super().__init__( self.MNI_Template,"MNI")
    def LoadMNIimage(self):
        MNI_Template=self.LoadTemplateImageFileMNI()
        return MNI_Template
    def LoadTemplateImageFileMNI(self):
        MNI_Template = sitk.ReadImage(self.MNI_Root)
        MNI_Template = sitk.DICOMOrient(MNI_Template, "LAI")
        MNI_Template = sitk.Cast(MNI_Template, sitk.sitkFloat32) 
        MNI_Template.SetDirection([-1,0,0, 0,-1,0, 0,0,-1])
        return MNI_Template
    def PrintINfO(self):
        print("Dimension:", self.MNI_Template.GetDimension())
        print("Size:", self.MNI_Template.GetSize())
        print("Spacing:", self.MNI_Template.GetSpacing())
        print("Origin:", self.MNI_Template.GetOrigin())
        print("Direction:", self.MNI_Template.GetDirection())
    def Load_png(self):
        img = plt.imread(self.MNI_Root)
        return img
class Image2D(ImageProcessor):
    def __init__(self,image,name):
        super().__init__(self)
        self.image=image
        self.name=name
    def GetSize(self):
        self.image.ConvertToNumpy()
        print(f"{self.name} shape:",  self.image.shape)
    def GetMinMax(self):
        image = self.ConvertToNumpy().flatten()
        print(f"{self.name} min:", image.min(), f"{self.name} max:", image.max())
    def MinMaxNormalize(self):
        image=self.ConvertToNumpy()
        image = image.astype(np.float32)

        min_val = np.min(image)
        max_val = np.max(image)

        normalized = (image- min_val) / (max_val - min_val + 1e-8)

        return normalized
    def GaussianNormalize(self):
        mage = sitk.RescaleIntensity(self.image, 0, 1)
        normalized = sitk.DiscreteGaussian(self.image, variance=1.0)
        # image=self.ConvertToNumpy()
        # image = image.astype(np.float32)

        # mean = np.mean(image)
        # std = np.std(image)
        # print(f"{self.name} mean:", image.mean(), f"{self.name} std:", image.std())
        # normalized = (image - mean) / (std + 1e-8)
        return normalized
    def ConvertToNumpy(self):
        return super().ConvertToNumpy() 
class PairImage2D():
    def __init__(self,mri_img, pet_img,index):
        self.mri=Image2D(mri_img,"MRI")
        self.pet=Image2D(pet_img,"PET")
        self.mriNumpy = self.mri.ConvertToNumpy()
        self.petNumpy = self.pet.ConvertToNumpy()
        self.index=index
    def GetMinMax(self):
        self.mri.GetMinMax()
        self.pet.GetMinMax()
    def Plot2DPairs(self):

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(self.mriNumpy, cmap="gray" ,  vmin=0,        vmax=1)
        plt.title(f"MRI")
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(self.petNumpy, cmap="gray"  , vmin=0,        vmax=1)
        plt.title(f"PET")
        plt.axis("off")

        plt.suptitle(f"Plot 2D Pairs Subject({self.index+1})")

        plt.figure(figsize=(10, 5))
        plt.show()
    def PlotHist(self):

        mri=self.mriNumpy.flatten()  #MRI
        pet=self.petNumpy.flatten()  #PET

        # optional cleanup (recommended)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(mri, bins=100)
        plt.title("MRI")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel Count")

        plt.subplot(1, 2, 2)
        plt.hist(pet, bins=100)
        plt.title("PET")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel Count")

        plt.suptitle(f"Subject {self.index+1}")
        plt.show()
    def PlotHist2(self):

        mri=self.mriNumpy.flatten()  #MRI
        pet=self.petNumpy.flatten()  #PET

        counts_mri, bins_mri = np.histogram(mri, bins=100)
        counts_pet, bins_pet = np.histogram(pet, bins=100)

        threshold = 400

        mask_mri = counts_mri <= threshold
        mask_pet = counts_pet <= threshold

        centers_mri = (bins_mri[:-1] + bins_mri[1:]) / 2
        centers_pet = (bins_pet[:-1] + bins_pet[1:]) / 2

        width_mri = bins_mri[1] - bins_mri[0]
        width_pet = bins_pet[1] - bins_pet[0]

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.bar(
            centers_mri[mask_mri],
            counts_mri[mask_mri],
            width=width_mri
        )
        plt.title("MRI")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel Count")

        plt.subplot(1,2,2)
        plt.bar(
            centers_pet[mask_pet],
            counts_pet[mask_pet],
            width=width_pet
        )
        plt.title("PET")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel Count")

        plt.suptitle(f"Subject {self.index+1}")
        plt.tight_layout()
        plt.show()          
    def MinMaxNormPairs(self) :
        self.mriNumpy = self.mri.MinMaxNormalize()
        self.petNumpy = self.pet.MinMaxNormalize()
    def GaussianNormPairs(self) :
        self.mriNumpy = self.mri.GaussianNormalize()
        self.petNumpy = self.pet.GaussianNormalize()
    def SetPetInfoIntoMri(self):
        self.mri.SetOrigin(self.pet.GetOrigin())
        self.mri.SetSpacing(self.pet.GetSpacing())
        self.mri.SetDirection(self.pet.GetDirection())
    def GetPairsSize(self):
        self.mri.GetSize()
        self.pet.GetSize()
    def MaxBinSetOne(self):
        self.MaxBinSet(1)
    def MaxBinSetZero(self):
        self.MaxBinSet(0)
    def MaxBinSet(self,setValue):
        self.mriNumpy= self.CalculateMaxBinFromHistogram(self.mri,setValue)
        self.petNumpy= self.CalculateMaxBinFromHistogram(self.pet,setValue)
    def CalculateMaxBinFromHistogram(self, image, setValue):

        threshold = 400
        # Convert input to NumPy
        if isinstance(image, Image2D):
            image = image.ConvertToNumpy()

        elif isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)

        else:
            image = np.asarray(image)

        # Make sure numeric
        image = np.asarray(image, dtype=np.float32)
        image[image < 0.005] = setValue
        # Remove NaN / Inf for histogram calculation
        finite_mask = np.isfinite(image)
        data = image[finite_mask]

        if data.size == 0:
            return image.copy()

        # Histogram
        hist, bin_edges = np.histogram(data, bins=256)

        # Find ALL bins with count > 400
        valid_bins = np.where(hist > threshold)[0]

        if len(valid_bins) == 0:
            print("No bins have count >", threshold)
            return image.copy()

        # print("Valid bins:", valid_bins)
        # print("Number of valid bins:", len(valid_bins))

        # Create result
        result = image.copy()
       
        # Mask for pixels belonging to valid bins
        valid_pixel_mask = np.zeros(image.shape, dtype=bool)

        for bin_index in valid_bins:

            lower = bin_edges[bin_index]
            upper = bin_edges[bin_index + 1]

            # Last bin needs <=
            if bin_index == len(hist) - 1:
                bin_mask = (
                    (image >= lower) &
                    (image <= upper)
                )
            else:
                bin_mask = (
                    (image >= lower) &
                    (image < upper)
                )

            valid_pixel_mask |= bin_mask

        # Set ALL pixels from valid bins to 1
        if setValue == 1:
            result[valid_pixel_mask] = 1

        else:
            result[valid_pixel_mask] = 0

        # Keep NaN/Inf unchanged or choose another behavior
        return result






























































