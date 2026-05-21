import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F

torch.manual_seed(111)  # Random Generator Seed
#---------------------------------------------------------------------------
def Train_Test_Split_Dataset(dataset,percent):
    dataset_size = len(dataset)
    train_size = int(percent * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    

    print(f"Total samples: {dataset_size}")
    print(f"Train samples: {train_size}")
    print(f"Test  samples: {test_size}")
    return train_dataset, test_dataset
def Train_Test_DataLoader(train_dataset, test_dataset,batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
#---------------------------------------------------------------------------
def Device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    return device
#---------------------------------------------------------------------------
def Calculate_min_max_Image(image, name):
  
    
    img_min = image.min()
    img_max = image.max()
    
    print(f"{name}:\t min={img_min:.3f}, max={img_max:.3f}")
#---------------------------------------------------------------------------
def Transform():
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128)),            #Original image  512*512   ->    Resize to 128*128
                transforms.Normalize((0.5,), (0.5,))      #Original image 0<pix<+1   ->   -1<pix<+1
            ])
    return transform
#---------------------------------------------------------------------------
def Show(generator,epoch, train_dataset,device):
    mri_img, real_pet_img = train_dataset[0]
    # اضافه کردن batch dimension
    mri_img = mri_img.unsqueeze(0).to(device)
    real_pet_img = real_pet_img.unsqueeze(0).to(device)

    generator.eval()
    with torch.no_grad():
        generated_pet = generator(mri_img)

    # انتقال به CPU
    mri_img = mri_img.cpu()
    real_pet_img = real_pet_img.cpu()
    generated_pet = generated_pet.cpu()

    # نمایش تصاویر
    plt.figure(figsize=(12,4))

    # MRI Input
    plt.subplot(1,3,1)
    plt.title("MRI Input")
    plt.imshow(mri_img[0][0], cmap="gray")
    plt.axis("off")

    # Real PET
    plt.subplot(1,3,2)
    plt.title("Real PET")
    plt.imshow(real_pet_img[0][0], cmap="gray")
    plt.axis("off")

    # Generated PET
    plt.subplot(1,3,3)
    plt.title("Generated PET")
    plt.imshow(generated_pet[0][0], cmap="gray")
    plt.axis("off")

    plt.suptitle(f"After Epoch {epoch}", fontsize=16, color='blue')
    plt.show()
#---------------------------------------------------------------------------
def Convert_to_numpy(image):
     # ✔ SimpleITK → numpy
    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)

    # ✔ torch → numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return image
#---------------------------------------------------------------------------
def Show_Loss(losses_g,losses_d):
    plt.figure(figsize=(8,5))
    plt.plot(losses_g, label="Generator Loss", color="red")
    plt.plot(losses_d, label="Discriminator Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    #---------------------------------------------------------------------------
def Plot_Sample_of_Dataset(train_dataset):
    # One Sample of Dataset
    sample = train_dataset[0]
    mri_img = sample["mri"]
    pet_img = sample["pet"]

    # image: numpy array یا torch tensor
    mri_img=Convert_to_numpy(mri_img)
    pet_img=Convert_to_numpy(pet_img)

    Calculate_min_max_Image(mri_img,"MRI Input")
    Calculate_min_max_Image(pet_img,"Real PET")


    # Convert from numpy to Tensor and Delete Extra Channel
    mri_np = mri_img.squeeze()
    pet_np = pet_img.squeeze()

    # رسم کنار هم
    plt.figure(figsize=(10,5))  # اندازه تصویر کلی
    plt.subplot(1, 2, 1)        # 1 ردیف، 2 ستون، تصویر اول
    plt.imshow(mri_np, cmap="gray")
    plt.title("MRI")
    plt.axis()

    plt.subplot(1, 2, 2)        # تصویر دوم
    plt.imshow(pet_np, cmap="gray")
    plt.title("PET")
    plt.axis()

    plt.show()
#---------------------------------------------------------------------------
class PairedMedicalDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        root_dir structure:
        root_dir/
            CT/
                Patient_01/
                Patient_18/
            T1-MRI/
                Patient_01/
                Patient_18/
        """
        super().__init__()
        self.transform = transform
        self.ct_root = os.path.join(root_dir, "CT")
        self.mri_root = os.path.join(root_dir, "T1-MRI")

        # List all patient folders
        self.patients = sorted(os.listdir(self.ct_root))

        # Collect all file pairs
        self.pairs = []
        for patient in self.patients:
            ct_patient_dir = os.path.join(self.ct_root, patient)
            mri_patient_dir = os.path.join(self.mri_root, patient)

            ct_files = sorted(os.listdir(ct_patient_dir))
            mri_files = sorted(os.listdir(mri_patient_dir))

            assert len(ct_files) == len(mri_files), f"Files mismatch for {patient}"

            for ct_file, mri_file in zip(ct_files, mri_files):
                self.pairs.append((
                    os.path.join(ct_patient_dir, ct_file),
                    os.path.join(mri_patient_dir, mri_file)
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mri_path = self.pairs[idx]

        ct_img = Image.open(ct_path).convert("L")
        mri_img = Image.open(mri_path).convert("L")

        if self.transform:
            ct_img = self.transform(ct_img)
            mri_img = self.transform(mri_img)
        else:
            ct_img = self.default_transform(ct_img)
            mri_img = self.default_transform(mri_img)

        return mri_img, ct_img  # (input, target) format for GANs
#---------------------------------------------------------------------------
def lambda_lr(epoch):
    if epoch < 200:
        return 1.0  # lr ثابت
    else:
        return 1 - (epoch - 200) / 200  # خطی کاهش از 1 به 0 در 200 epoch بعدی
#---------------------------------------------------------------------------
def resize_2d_cwh(img, size=(256, 256)):

    # img: (C, H, W)

    if img.dim() == 2:
        img = img.unsqueeze(0)

    img = img.unsqueeze(0)  # (1, C, H, W)

    img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)

    return img.squeeze(0)
#---------------------------------------------------------------------------
def load_nii_files_fromDataset(niiDatasetRoot):
    samples = []
    for subject in os.listdir(niiDatasetRoot):
       
        subject_path = os.path.join(niiDatasetRoot, subject)

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
            # 🔥 convert to numpy
            mri = sitk.GetArrayFromImage(mri).astype(np.float32)
            pet = sitk.GetArrayFromImage(pet).astype(np.float32)
        
            mri = torch.as_tensor(mri).unsqueeze(0).float()
            pet = torch.as_tensor(pet).unsqueeze(0).float()
            mri = resize_2d_cwh(mri, (256, 256))
            pet = resize_2d_cwh(pet, (256, 256))
            # 🔥 store sample
            samples.append((mri, pet))

    return samples