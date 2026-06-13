import os
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
np.set_printoptions(threshold=np.inf)

class DatasetExplorer():

    def __init__(self,root_folder,dim):
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
    def FindPairFolders(self):
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
                    if fmt:
                        mri_dict[root] = fmt

                # ---------- PET ----------
                for root, _, _ in os.walk(pet_path):
                    fmt = self.DetectFileExtension(root)
                    if fmt:
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
                        if fmt:
                            pet_dict[pet_path] = fmt
                        fmt = self.DetectFileExtension(mri_path)
                        if fmt:
                            mri_dict[mri_path] = fmt


        return mri_dict,pet_dict
    def GetSubjects(self):
        subjects= os.listdir(self.subject)
        print("Total subjects:", len(subjects))
        return subjects
    def FindAllFileTypes(self):
        extensions = {}

        for root, _, files in os.walk(self.pet_root):
            for f in files:
                ext = os.path.splitext(f)[1].lower()

                if ext == "":
                    ext = "NO_EXTENSION"

                extensions[ext] = extensions.get(ext, 0) + 1

        print("File types:")
        for k, v in sorted(extensions.items(), key=lambda x: -x[1]):
            print(k, ":", v)

        return extensions
    def CountDifferentFileTypes(self):
        dicom_folders = set()
        v_folders = set()
        i_folders = set()
        for root, _, files in os.walk(self.pet_root):

            has_dicom = False
            has_v = False
            has_i = False
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

            # One time for each Folder
            if has_dicom:
                dicom_folders.add(root)

            if has_v:
                v_folders.add(root)

            if has_i:
                i_folders.add(root)

        print("Results:")
        print("DICOM folders:", len(dicom_folders))
        print("V folders:", len(v_folders))
        print("i folders:", len(i_folders))
        print("Total medical folders:", len(dicom_folders) + len(v_folders)+ len(i_folders))
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
    def IsFileExtension(self,extension,files):
        return any(f.lower().endswith(extension) for f in files)
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

        return None
    
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
    def ConvertToNumpy(img):
        if img is None:
            raise ValueError("Image is None")

        if isinstance(img, np.ndarray):
            return img

        return sitk.GetArrayFromImage(img)

class DatasetBuilder():
    def __init__(self,input_nii,output_2D_nii_Dataset):
        self.input_nii = input_nii
        self.output_2D_nii_Dataset = output_2D_nii_Dataset   
    def Create_dataset_nii(self,mri_np,pet_np,subject,count):
        affine = np.eye(4)

        mri_nii = nib.Nifti1Image(mri_np, affine)
        pet_nii = nib.Nifti1Image(pet_np, affine)

        subject_out = os.path.join(self.root_folder, subject)
        pair_out = os.path.join(subject_out, f"pair_{count:02d}")

        mri_out_dir = os.path.join(pair_out, "mri")
        pet_out_dir = os.path.join(pair_out, "pet")

        os.makedirs(mri_out_dir, exist_ok=True)
        os.makedirs(pet_out_dir, exist_ok=True)

        mri_path = os.path.join(mri_out_dir, "mri.nii.gz")
        pet_path = os.path.join(pet_out_dir, "pet.nii.gz")

        nib.save(mri_nii, mri_path)
        nib.save(pet_nii, pet_path)
    def Create_3D_nii_Dataset(self):

        MNI_Template=self.Load_MNI_Template_Image_File(self.MNI_Root)  

        os.makedirs(self.output_dir, exist_ok=True)
        totalPairImagesCount = 0

        subjects = os.listdir(self.mri_root)
        print("Total subjects:", len(subjects))

        for i, subject in enumerate(subjects):

            print("Subject",i+1,":", subject)

            eachSubjectPairImagesCount=1

            mri_path = os.path.join(self.mri_root, subject)
            pet_path = os.path.join(self.pet_root, subject)

            if not os.path.isdir(mri_path):
                continue
            if not os.path.exists(pet_path):
                continue

            if os.path.isdir(mri_path) and os.path.exists(pet_path):

                mri_folders = []
                pet_folders = []

                # collect all MRI folders
                for d, _, _ in os.walk(mri_path):
                    fmt = self.detect_format(d)
                    if fmt:
                        mri_folders.append(d)

                # collect all PET folders
                for d, _, _ in os.walk(pet_path):
                    fmt = self.detect_format(d)
                    if fmt:
                        pet_folders.append(d)

                # loop over pairs
                for idx,(mri_folder, pet_folder) in enumerate(zip(mri_folders, pet_folders)):
                    
                    mri_fmt = self.detect_format(mri_folder)
                    pet_fmt = self.detect_format(pet_folder)
                    mri, mri_frames = self.Load_Images(mri_folder, mri_fmt,"mri")
                    pet, pet_frames = self.Load_Images(pet_folder, pet_fmt,"pet")

                    subject_out = os.path.join(self.output_dir, subject)
                    os.makedirs(subject_out, exist_ok=True)

                    pair_out = os.path.join(subject_out, f"pair_{idx+1}")
                    os.makedirs(pair_out, exist_ok=True)

                    # ✔
                    mri= self.register_pet_to_mri(mri, MNI_Template)
                    pet= self.register_pet_to_mri(pet, mri)

                    # ✔ save correctly
                    sitk.WriteImage(mri, os.path.join(pair_out, "mri.nii.gz"))
                    sitk.WriteImage(pet, os.path.join(pair_out, "pet.nii.gz"))

                    print(eachSubjectPairImagesCount)
                    eachSubjectPairImagesCount+=1
                totalPairImagesCount += 1
        print ("Total Pair Images:",totalPairImagesCount) 
    def Create_2D_nii_Dataset(self,sliceId):
        totalPairImagesCount=0
        for subjectCounter,subject in  enumerate(os.listdir(self.input_nii)):
            print("Subject",subjectCounter+1,":", subject)

            subject_path = os.path.join(self.input_nii, subject)

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

                # ✔ read images
                mri = self.Load_NIfTI_Image_File(mri_path)
                pet = self.Load_NIfTI_Image_File(pet_path)

                subject_out = os.path.join(self.output_2D_nii_Dataset, subject)
                os.makedirs(subject_out, exist_ok=True)

                pair_out = os.path.join(subject_out, f"pair_{pairCounter+1}")
                os.makedirs(pair_out, exist_ok=True)
                mri_s = self.get_slice_Id(mri, sliceId)
                pet_s = self.get_slice_Id(pet, sliceId)

                #plot_pair_2D(mri_s, pet_s)

                # ✔ save correctly
                sitk.WriteImage(mri_s, os.path.join(pair_out, "mri.nii.gz"))
                sitk.WriteImage(pet_s, os.path.join(pair_out, "pet.nii.gz"))

                print(pairCounter+1)
                pairCounter+=1
            totalPairImagesCount += 1
        print("Total Pair Images:",totalPairImagesCount) 
    def Create_2D_Normalized_Filtered_nii_Dataset(self):
        totalPairImagesCount=0
        for subjectCounter,subject in  enumerate(os.listdir(self.input_nii)):
            print("Subject",subjectCounter+1,":", subject)

            subject_path = os.path.join(self.input_nii, subject)

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

                # ✔ read images
                mri = self.Load_NIfTI_Image_File(mri_path)
                pet = self.Load_NIfTI_Image_File(pet_path)
                mri=self.minmax_normalize(mri)
                pet=self.minmax_normalize(pet)
                mri_masked, pet_masked=self.Max_bin_Set_One(mri, pet)

                subject_out = os.path.join(self.output_2D_nii_Dataset, subject)
                os.makedirs(subject_out, exist_ok=True)

                pair_out = os.path.join(subject_out, f"pair_{pairCounter+1}")
                os.makedirs(pair_out, exist_ok=True)

                self.plot_pair_2D(mri_masked, pet_masked)

                # ✔ save correctly
                sitk.WriteImage(mri_masked, os.path.join(pair_out, "mri.nii.gz"))
                sitk.WriteImage(pet_masked, os.path.join(pair_out, "pet.nii.gz"))

                print(pairCounter+1)
                pairCounter+=1
            totalPairImagesCount += 1
        print("Total Pair Images:",totalPairImagesCount) 

class ImageLoader():
    def __init__(self,mri_dict, pet_dict):
        self.mri_dict = mri_dict
        self.pet_dict = pet_dict
    
    def Load_Dataset(self,number=None):
        dataset = []

        items = list(zip(self.mri_dict.items(), self.pet_dict.items()))
        if number is not None:
            items = items[:number]

        for (mri_folder, mri_fmt), (pet_folder, pet_fmt) in items:
         
            # print("MRI Folder:", mri_folder)
            # print("MRI Format:", mri_fmt)

            # print("PET Folder:", pet_folder)
            # print("PET Format:", pet_fmt)

            mri, mri_frames = self.Load_Images(mri_folder, mri_fmt, "mri")
            pet, pet_frames = self.Load_Images(pet_folder, pet_fmt, "pet")
            dataset.append((mri, pet))
        return dataset
    def Load_Images(self,folder, fmt, mode):
        if fmt == ".dicom":
            return self.Load_dicom_file(folder, mode)
        elif fmt == ".v":
            return self.Load_v_file(folder, mode)
        elif fmt == ".i":
            return self.Load_i_file(folder, mode)
        elif fmt == ".gz":
            return self.Load_NIfTI_Image_File(folder, mode)
        else:
            raise ValueError("Unknown format")
    def Load_dicom_file(self,folder, mode):
        frames=1
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder)
        n = len(dicom_names)
        if n == 0:
            return None

        if mode=="pet" and self.IsDynamicPETImage(folder):
            dicom_names = dicom_names[:n // 6]
            frames=6
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except:
            return None
        
        #image=crop_empty_space(image)
        image = sitk.DICOMOrient(image, "RAI")           #Orientation : RAS   #LPS
        
        image=self.ConvertImageToFloat32(image)     #Type conversion
   
        image = sitk.RescaleIntensity(image, 0, 1)

        #image = sitk.Median(image, [1, 1, 1])            #Filter denoising
        image = sitk.DiscreteGaussian(image, variance=1.0)
        original = image
        arr = sitk.GetArrayFromImage(image)              #Normalization
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) 
        arr = np.where(arr < 0.01, 0, arr)
        image = sitk.GetImageFromArray(arr)
        image.CopyInformation(original)
        if mode=="pet" and (image.GetSize()==(128, 128, 47) or image.GetSize()==(400, 400, 109) or image.GetSize()==(128, 128, 31)):
            image = sitk.DICOMOrient(image, "LAS") 
            image.SetDirection([-1,0,0, 0,-1,0, 0,0,-1]) 
        return (image,frames)
    def Load_v_file(self, folder, mode="",frames=6, slices=63, rows=128, cols=128, header=0):
        v_files = []
        for root, _, files in os.walk(folder):
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
        arr = np.flip(arr, axis=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) 
        arr = np.where(arr < 0.01, 0, arr)
        
        # convert to SimpleITK
        image = sitk.GetImageFromArray(arr)
        image = sitk.DICOMOrient(image, "RPS")
        if "30_min_3D_FDG_4i_16s" in v_files[0]:
            image.SetSpacing((2.1, 2.1, 2.4))
        else:
            image.SetSpacing((2.6, 2.6, 2.4))
        image.SetOrigin((128.0, 128.0, 75.6))
        image.SetDirection([-1,0,0, 0,-1,0,0,0,-1])
        return (image,frames)
    def Load_i_file(self, folder, mode="",dtype=np.float32, slices=207, rows=256,cols=256,header=0):
        i_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".i"):
                    i_files.append(os.path.join(root, f))

        if len(i_files) == 0:
            raise FileNotFoundError("No .i file found")
        
        # read data
        file_path = i_files[0]
        data = np.fromfile(file_path, dtype=dtype, offset=header)
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
        image = sitk.DICOMOrient(image, "RPI")
        image.SetSpacing((1.2, 1.2,1.2))
        image.SetOrigin((128.0, 128.0, 75.6))
        image.SetDirection([-1,0,0,
                    0,-1,0,
                    0,0,-1])
        return (image,frames)
    def IsDynamicPETImage(self,folder):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder)

        reader.SetFileNames(dicom_names)

        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        reader.Execute()

        try:
            num_frames = int(reader.GetMetaData(0, "0054|0101"))  # Number of Time Slices
            return num_frames > 1
        except:
            return False
    
    def Load_png(self,MNI_Root):

        img = plt.imread(self.root_folder)
        return img
    def Load_MNI_Template_Image_File(MNI_Root):
        MNI_Template = sitk.ReadImage(MNI_Root)
        MNI_Template = sitk.DICOMOrient(MNI_Template, "LAI")
        MNI_Template = sitk.Cast(MNI_Template, sitk.sitkFloat32) 
        MNI_Template.SetDirection([-1,0,0, 0,-1,0, 0,0,-1])
        return MNI_Template
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
    def Load_NIfTI_Image_File(self,folder, mode):
        img = sitk.ReadImage(folder)
        if img.GetDimension() == 3:
            img = sitk.DICOMOrient(img, "RAI")
            img.SetDirection([-1,0,0, 0,-1,0, 0,0,-1])
        else:
            img.SetDirection([-1,0, 0,-1])
        img = sitk.Cast(img, sitk.sitkFloat32) 
        
        return img, 1
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
    def ConvertImageToFloat32(self,image):
        image = sitk.Cast(image, sitk.sitkFloat32)
        return image

class Preprocessor():
    def __init__(self,dataset):
        self.dataset=dataset  
    def Dataset2DSliceId(self,sliceId=None):
        new_dataset = []
        for mri, pet in self.dataset:
            mri=self.ConvertToNumpy(mri)
            pet=self.ConvertToNumpy(pet)

            if sliceId is None:
                mri2D,mriSliceId=self.GetMidSlice(mri)
                pet2D,petSliceId=self.GetMidSlice(pet)
            else:
                mri2D=self.GetSliceId(mri,sliceId)
                pet2D=self.GetSliceId(pet,sliceId)

            new_dataset.append((mri2D, pet2D))   

            dataset2D = new_dataset
        return dataset2D
    def RegisteredDataset(self):
        new_dataset = []

        for mri, pet in self.dataset:
            pet_reg = self.register_pet_to_mri(mri, pet)
            new_dataset.append((mri, pet_reg))

        self.dataset = new_dataset
    def register_pet_to_mri(self,MRI,PET):
        
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
    def GetMidSlice(self,image):
        mid = image.shape[0] // 2
        sl = image[mid]
        return sl, mid   
    def GetSliceId(self,image,sliceId):
        if sliceId < 0 or sliceId >= image.shape[0]:
            raise ValueError("sliceId out of range")
        return image[sliceId]
    def ConvertToNumpy(self,img):
        if img is None:
            raise ValueError("Image is None")

        if isinstance(img, np.ndarray):
            return img

        return sitk.GetArrayFromImage(img)
   
class ImageInfo():
    def __init__(self,image,type=""):
        self.image=image
        self.type=type
    def Image_Info(self):
        self.GetOrientartion()
        self.GetOrigin()
        self.GetSpacing()
        self.GetDirection()     
    def GetOrientartion(self):
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(self.image.GetDirection())
        print("Detected "+self.type+" Image Orientartion:", orientation)                                                                
    def GetOrigin(self):
        origin= self.image.GetOrigin()
        print("Detected "+self.type+" Image Origin:", origin)
    def GetSpacing(self):
        spacing=self.image.GetSpacing()
        print("Detected "+self.type+" Image Spacing:", spacing)
    def GetDirection(self):
        direction=self.image.GetDirection()
        print("Detected "+self.type+" Image Direction:",direction)
   
    def minmax_normalize(self):
        # convert if needed
        img = self.ConertToNumpy(self.image)
        img = img.astype(np.float32)

        min_val = np.min(img)
        max_val = np.max(img)

        normalized = (img - min_val) / (max_val - min_val + 1e-8)

        return normalized
    def gaussian_normalize(self):
        # convert if needed
        img = self.ConvertToNumpy(self.image)
        img = img.astype(np.float32)

        mean = np.mean(img)
        std = np.std(img)

        normalized = (img - mean) / (std + 1e-8)

        return normalized

    def filter(self):
        image = sitk.Median(self.image, [3, 3, 3]) 
        image = sitk.DiscreteGaussian(image, variance=1.0)
        return image
    def GetDicomImageSize(self):

     
        # optional but recommended
        image = sitk.DICOMOrient( self.image, "LPS")

        # convert to numpy (z, y, x)
        arr = self.ConvertToNumpy(image)

        # number of slices
        z = arr.shape[0]
        y = arr.shape[1]
        x = arr.shape[2]

        print("Shape (z, y, x):", arr.shape)
        print("Number of slices (z):", z)

        return image, arr
    
class Visualizer():
    def __init__(self,dataset):
        self.dataset=dataset   
    def Plot2DPairs(self,index=None):
        data = self.dataset if index is None else self.dataset[:index]

        for i,(mri_img, pet_img) in enumerate(data):
            # FORCE conversion ONLY ONCE
            mri=self.ConvertToNumpy(mri_img)  #MRI
            pet=self.ConvertToNumpy(pet_img)  #PET
            print("MRI shape:", mri.shape)
            print("PET shape:", pet.shape)

            plt.figure(figsize=(10,5))

            plt.subplot(1,2,1)
            plt.imshow(mri, cmap="gray" )
            plt.title(f"MRI")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.imshow(pet, cmap="gray" )
            plt.title(f"PET")
            plt.axis("off")

            plt.suptitle(f"Plot 2D Pairs Subject({i+1})")
            plt.show()
    def plotThreeViewsMNITemplate(self,MNI_Root):
        MNI_Template=self.Load_MNI_Template_Image_File(MNI_Root)
        self.show_three_views(MNI_Template)
        print("Dimension:", MNI_Template.GetDimension())
        print("Size:", MNI_Template.GetSize())
        print("Spacing:", MNI_Template.GetSpacing())
        print("Origin:", MNI_Template.GetOrigin())
        print("Direction:", MNI_Template.GetDirection())
    def PlotHist(self):

        plt.hist(self.Image, bins=100)

        plt.title(" Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel Count")
        plt.show()
        counts, bins = np.histogram(self.image, bins=100)
        max_bin_idx = np.argmax(counts)
        #left_idx = max(0, max_bin_idx - 4)
        #right_idx = min(len(bins)-1, max_bin_idx + 5)
        #bin_min = bins[left_idx]
        #bin_max = bins[right_idx]
        #mask = (Image >= bin_min) & (Image < bin_max)
        #Image[mask] = 1
        #arr = sitk.GetArrayFromImage(Image).copy()

    

        #kernel = np.ones((2,2))

        #count = convolve(arr.astype(np.int32), kernel, mode='constant')

        #filtered = (count > 4).astype(np.uint8)
        #return Image
    def PlotMiddleSlice(self, title=""):
        print("ITK size:", self.image.GetSize())

        arr = self.ConvertToNumpy(self.image)
        arr = np.transpose(arr, (1, 2, 0))
        mid_slice = arr[arr.shape[0] // 2]
        
        print(arr.shape)
        plt.imshow(mid_slice, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
    def show_three_views(self,vol,subject="",subjectId="",pair=""):
        vol = self.convert_to_NumPy(vol)
        z, y, x = vol.shape
        print(vol.shape)
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(vol[z//2], cmap='gray')
        plt.title("Axial")

        plt.subplot(1,3,2)
        plt.imshow(vol[:, y//2, :], cmap='gray')
        plt.title("Coronal")

        plt.subplot(1,3,3)
        plt.imshow(vol[:, :, x//2], cmap='gray')
        plt.title("Sagittal")

        plt.suptitle(f"Subject({subject}):{subjectId}, Pair({pair})")  

        plt.show()
    def PlotPairsSliceID(self, sliceId,title=""):

        # FORCE conversion ONLY ONCE
        mri=self.convert_to_NumPy(self.MRI)
        pet=self.convert_to_NumPy(self.PET)

        print("MRI shape:", mri.shape)
        print("PET shape:", pet.shape)

        mri_s, mri_idx = self.get_slice(mri, sliceId)
        pet_s, pet_idx = self.get_slice(pet, sliceId)
        
    

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(mri_s, cmap="gray" )
        plt.title(f"MRI (slice {mri_idx})")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(pet_s, cmap="gray" )
        plt.title(f"PET (slice {pet_idx})")
        plt.axis("off")

        plt.suptitle(title)
        plt.show()
    def HistofPairs(self):


        # convert to numpy if needed
        mri = self.ConvertToNumpy(self.MRI)
        pet = self.ConvertToNumpy(self.PET)

        mri_flat = mri.flatten()
        pet_flat = pet.flatten()


        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        # MRI Histogram
        axes[0].hist(mri.flatten(), bins=100)

        axes[0].set_title("MRI Histogram")
        axes[0].set_xlabel("Intensity")
        axes[0].set_ylabel("Voxel Count")

        # PET Histogram
        axes[1].hist(pet.flatten(), bins=100)

        axes[1].set_title("PET Histogram")
        axes[1].set_xlabel("Intensity")
        axes[1].set_ylabel("Voxel Count")

        plt.tight_layout()
        plt.show()
    def ConvertToNumpy(self,img):
        if img is None:
            raise ValueError("Image is None")

        if isinstance(img, np.ndarray):
            return img

        return sitk.GetArrayFromImage(img)
    def Max_bin_Set_One(self, bins=100):

        # convert to numpy if needed
        mri = self.ConvertTONumpy(self.MRI)
        pet = self.ConvertTONumpy(self.PET)

        mri_flat = mri.flatten()
        pet_flat = pet.flatten()


        # histogram MRI
        mri_counts, mri_bins = np.histogram(mri_flat, bins=bins)
        peak = np.argmax(mri_counts)

        left = max(0, 0)
        right = min(len(mri_counts) - 1, peak + 1)

        mri_min = mri_bins[left]
        mri_max = mri_bins[right + 1]

        mri_masked = mri.copy()
        mri_masked[(mri >= mri_min) & (mri < mri_max)] = 0

        # histogram PET
        pet_counts, pet_bins = np.histogram(pet_flat, bins=bins)
        pet_max_bin = np.argmax(pet_counts)

        pet_min = pet_bins[pet_max_bin]
        pet_max = pet_bins[pet_max_bin + 1]

        pet_masked = pet.copy()
        pet_masked[(pet >= pet_min) & (pet < pet_max)] = 0


        return mri_masked, pet_masked
    def Set_Image_Info(self):
        self.MRI.SetOrigin(self.PET.GetOrigin())
        self.MRI.SetSpacing(self.PET.GetSpacing())
        self.MRI.SetDirection(self.PET.GetDirection())
    









































































