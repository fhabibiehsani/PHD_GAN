import io
import os
import cv2
import lpips
import torch
import random
import shutil
import datetime
import platform
import warnings
import contextlib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from piq import ssim, multi_scale_ssim
from torch import nn
from torch_fidelity import calculate_metrics
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=UserWarning)
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
#---------------------------------------------------------------------------
def Train_Test_DataLoader(train_dataset, test_dataset,batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader,test_loader
#---------------------------------------------------------------------------
def GetDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
#---------------------------------------------------------------------------
def GetAndPrintDeviceInfo():
    device = GetDevice()
    print("Device:", device)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    return device
#---------------------------------------------------------------------------
def Transform(W,H):
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((W,H)),                 #Original image  512*512   ->    Resize to 128*128
                transforms.Normalize((0.5,), (0.5,))      #Original image 0<pix<+1   ->   -1<pix<+1
            ])
    return transform
#---------------------------------------------------------------------------
def Calculate_min_max_Image(image, name):
    img_min = image.min()
    img_max = image.max()
    
    print(f"{name}:\t min={img_min:.3f}, max={img_max:.3f}")
#---------------------------------------------------------------------------
def Show(generator,epoch, test_loader,device,save_path):
    mri_img, real_pet_img = next(iter(test_loader))

    # keep batch dimension
    mri_img = mri_img.to(device)
    real_pet_img = real_pet_img.to(device)

    generator.eval()

    with torch.no_grad():
        generated_pet = generator(mri_img)

    # only show first image
    mri_img = mri_img[0:1].cpu()
    real_pet_img = real_pet_img[0:1].cpu()
    generated_pet = generated_pet[0:1].cpu()
    Calculate_min_max_Image(real_pet_img,"real_pet_img")
    Calculate_min_max_Image(generated_pet,"generated_pet")
    # نمایش تصاویر
    plt.figure(figsize=(12,5))

    # MRI Input
    plt.subplot(1,5,1)
    plt.title("Input MRI")
    plt.imshow(mri_img[0][0], cmap="gray")
    plt.axis("off")

    # Real PET
    plt.subplot(1,5,2)
    plt.title("Real PET")
    plt.imshow(real_pet_img[0][0], cmap="gray")
    plt.axis("off")

    # Generated PET
    plt.subplot(1,5,3)
    plt.title("Generated PET")
    plt.imshow(generated_pet[0][0], cmap="gray")
    plt.axis("off")

    # Generated PET -Real PET
    plt.subplot(1,5,4)

    plt.title("Generated PET-Real PET")
    diff = generated_pet[0, 0].detach().cpu() - real_pet_img[0, 0].detach().cpu()
    plt.imshow(diff , cmap="seismic", vmin=-1, vmax=1)
    #plt.colorbar()
    plt.axis("off")
    # real, fake: [1,1,H,W] or [1,3,H,W]

    error = torch.abs(real_pet_img[0][0]-generated_pet[0][0])
    plt.subplot(1,5,5)
    plt.imshow(error.cpu(), cmap='hot')
    plt.title("Absolute Error")
    #plt.colorbar()
    plt.axis("off")

    plt.suptitle(f"After Epoch {epoch}", fontsize=16, color='black')
    # Save figure
    file_path = os.path.join(save_path,f"Image.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
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

    
    plt.figure(figsize=(10,5)) 
    plt.subplot(1, 2, 1)      
    plt.imshow(mri_np, cmap="gray")
    plt.title("MRI")
    plt.axis()

    plt.subplot(1, 2, 2)       
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
    constant = 20
    max_epoch = 100

    if epoch < constant:
        return 1.0
    else:
        return max(0.0, 1 - (epoch - constant) / (max_epoch - constant))  # خطی کاهش از 1 به 0 در 200 epoch بعدی
#---------------------------------------------------------------------------
def resize_2d_cwh(img, size=(256, 256)):

    # img: (C, H, W)

    if img.dim() == 2:
        img = img.unsqueeze(0)

    img = img.unsqueeze(0)  # (1, C, H, W)

    img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)

    return img.squeeze(0)
#---------------------------------------------------------------------------
    if img is None:
        raise ValueError("Image is None")

    if isinstance(img, np.ndarray):
        return img

    return sitk.GetArrayFromImage(img)
#---------------------------------------------------------------------------
def normalizeTanh(x):
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min )  # [0,1]
    x = x * 2.0 - 1.0                   # [-1,1]
    return x
#---------------------------------------------------------------------------
def Load_nii_Files_From_Dataset(niiDatasetRoot):
    samples = []
    for subject in sorted(os.listdir(niiDatasetRoot)):
       
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
            mri = normalizeTanh(mri)
            pet = normalizeTanh(pet)
        
            mri = torch.as_tensor(mri).unsqueeze(0).float()
            pet = torch.as_tensor(pet).unsqueeze(0).float()
            mri = resize_2d_cwh(mri, (256, 256))
            pet = resize_2d_cwh(pet, (256, 256))
            # 🔥 store sample
            samples.append((mri, pet))

    return samples
#---------------------------------------------------------------------------
def Load_nii_Files_From_Dataset2(niiDatasetRoot):
    samples = []
    for subject in  sorted(os.listdir(niiDatasetRoot)):
       
        subject_path = os.path.join(niiDatasetRoot, subject)

        if not os.path.isdir(subject_path):
            continue
        
        mri_path = os.path.join(subject_path, "mri.nii")
        pet_path = os.path.join(subject_path, "pet.nii")

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
        
        #mri = BiasFieldCorrectionN4(mri)
        # PET Dont Need

        #mri = MedianFilter(mri, size=5)
        #pet = MedianFilter(pet, size=5)

        mri = normalizeTanh(mri)
        pet = normalizeTanh(pet)
    
        mri = torch.as_tensor(mri).unsqueeze(0).float()
        pet = torch.as_tensor(pet).unsqueeze(0).float()

        mri = resize_2d_cwh(mri, (128, 128))
        pet = resize_2d_cwh(pet, (128, 128))
        # 🔥 store sample
        samples.append((mri, pet))

        #DataAgumentation

        # # 1. Rotation (applied to BOTH)
        # angle = random.uniform(-10, 10)
        # mri = TF.rotate(mri, angle)
        # pet = TF.rotate(pet, angle)
        # samples.append((mri, pet))

        # # 2. Horizontal flip (same decision)
        # mri = TF.hflip(mri)
        # pet = TF.hflip(pet)
        # samples.append((mri, pet))

        # # 3. Random vertical flip
        # mri = TF.vflip(mri)
        # pet = TF.vflip(pet)
        # samples.append((mri, pet))

        # # 4. Affine (translation + scaling)
        # translate = (0.05, 0.05)
        # scale = random.uniform(0.9, 1.1)
        # mri = TF.affine(mri, angle=0, translate=translate, scale=scale, shear=0)
        # pet = TF.affine(pet, angle=0, translate=translate, scale=scale, shear=0)
        # samples.append((mri, pet))
        #-------------

    return samples
#---------------------------------------------------------------------------
def plot_triplet(mri, real_pet, fake_pet):

    mri=Convert_to_numpy(mri)
    real_pet=Convert_to_numpy(real_pet)
    fake_pet=Convert_to_numpy(fake_pet)

    if mri.ndim == 4:
        mri = mri[0,0]
    if real_pet.ndim == 4:
        real_pet = real_pet[0,0]
    if fake_pet.ndim == 4:
        fake_pet = fake_pet[0,0]

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(mri, cmap='gray')
    axes[0].set_title("MRI")
    axes[0].axis("off")

    axes[1].imshow(real_pet, cmap='gray')
    axes[1].set_title("Real_PET")
    axes[1].axis("off")

    axes[2].imshow(fake_pet, cmap='gray')
    axes[2].set_title("Fake_PET")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------
def Plot_Example(generator, test_loader, device):
    generator.eval()

    with torch.no_grad():
        for mri,real_pet in test_loader:

            real_pet = real_pet.to(device).float()
            fake_pet = generator(mri)
            plot_triplet(mri, real_pet, fake_pet) 
#---------------------------------------------------------------------------
def MedianFilter(Image,size=5):
    filteredImage = cv2.medianBlur(Image, size)  # kernel size must be odd: 3,5,7,...
    return filteredImage
#---------------------------------------------------------------------------
def BiasFieldCorrectionN4(mri_array):

    img = sitk.GetImageFromArray(mri_array)
    # Convert to float (IMPORTANT)
    img = sitk.Cast(img, sitk.sitkFloat32)
    # Initial mask (foreground estimation)
    maskImage = sitk.OtsuThreshold(img, 0, 1, 200)
    # N4 Bias Field Correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    correctedImage = corrector.Execute(img, maskImage)
    # Convert back to numpy
    correctedArray = sitk.GetArrayFromImage(correctedImage)

    return correctedArray
#---------------------------------------------------------------------------
def DataAugmentations():
    #Rotation (small angles)
    transforms.RandomRotation(degrees=10)
    #Flip (if anatomically valid)
    transforms.RandomHorizontalFlip(p=0.5)
    transforms.RandomVerticalFlip(p=0.2)
    #Translation / shift
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
    #Scaling (zoom in/out)
    transforms.RandomResizedCrop(size=128, scale=(0.9, 1.1))
#---------------------------------------------------------------------------
def IntensityShift(x):
    factor = random.uniform(0.9, 1.1)
    return x * factor
#---------------------------------------------------------------------------
def AddNoise(x, sigma=0.02):
    noise = torch.randn_like(x) * sigma
    return x + noise
#---------------------------------------------------------------------------
def Train_UNet(num_epochs,train_loader,train_dataset,device,optimizer_generator,generator,l1_loss,l2_loss,ssim_loss,lambda_L1,lambda_L2,lambda_ssim,scheduler_G):
    best_loss = float("inf")

    losses = []

    for epoch in range(num_epochs):

        epoch_loss = 0

        for n, (MRI, real_PET) in enumerate(train_loader):

            MRI = MRI.to(device)
            real_PET = real_PET.to(device)

            optimizer_generator.zero_grad()

            # ---------------------
            # Forward (UNet)
            # ---------------------
            fake_PET = generator(MRI)

            # ---------------------
            # Losses (NO GAN LOSS)
            # ---------------------
            loss_l1 = l1_loss(fake_PET, real_PET) * lambda_L1
            loss_l2 = l2_loss(fake_PET, real_PET) * lambda_L2
            loss_ssim = ssim_loss(fake_PET, real_PET) * lambda_ssim

            loss = loss_l1 + loss_l2 + loss_ssim

            # ---------------------
            # Backprop
            # ---------------------
            loss.backward()
            optimizer_generator.step()

            epoch_loss += loss.item()

        # ---------------------
        # Scheduler step
        # ---------------------
        scheduler_G.step()

        losses.append(epoch_loss / len(train_loader))

        # ---------------------
        # Print every 10 epochs
        # ---------------------
        if (epoch + 1) % 10 == 0:

            print(f"Epoch: {epoch+1}")
            print(f"LR G: {scheduler_G.get_last_lr()[0]:.6f}")
            print(f"Loss: {losses[-1]}")

            Show(generator, epoch+1, train_dataset, device)
            Show_Loss(losses)
#---------------------------------------------------------------------------
def CalculateMeanMetrics(fold_results):
    mean_metrics = {}
    std_metrics = {}

    for key in fold_results[0].keys():

        values = [ fold[key] for fold in fold_results]

        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)


    # print("\n========== Final 5-Fold Results ==========")

    # for key in mean_metrics:
    #     print(f"{key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    return mean_metrics,std_metrics  #Todo
#---------------------------------------------------------------------------
def SaveExperimentInfo(
        filepath,
        model,
        hyperparameters,
        fold_results,
        mean_metrics=None,
        std_metrics=None,
        training_time=None,
        random_seeds=None
    ):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:

        f.write("="*60 + "\n")
        f.write("EXPERIMENT INFORMATION\n")
        f.write("="*60 + "\n\n")


        # -------------------------
        # Date and Environment
        # -------------------------
        f.write("Date:\n")
        f.write(str(datetime.datetime.now()) + "\n\n")


        f.write("Python Version:\n")
        f.write(platform.python_version() + "\n\n")


        f.write("PyTorch Version:\n")
        f.write(torch.__version__ + "\n\n")


        # -------------------------
        # Hardware Information
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Hardware Information\n")
        f.write("="*60 + "\n")


        if torch.cuda.is_available():

            gpu_name = torch.cuda.get_device_name(0)

            f.write(f"GPU: {gpu_name}\n")

            f.write(
                f"CUDA Version: {torch.version.cuda}\n"
            )

            f.write(
                f"GPU Total Memory: "
                f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB\n"
            )

            f.write(
                f"Allocated Memory: "
                f"{torch.cuda.memory_allocated()/1024**2:.2f} MB\n"
            )

            f.write(
                f"Reserved Memory: "
                f"{torch.cuda.memory_reserved()/1024**2:.2f} MB\n"
            )

            f.write(
                f"Max Allocated Memory: "
                f"{torch.cuda.max_memory_allocated()/1024**2:.2f} MB\n"
            )

        else:
            f.write("GPU: Not available\n")


        f.write("\n")


        # -------------------------
        # Model Information
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Model Information\n")
        f.write("="*60 + "\n")


        f.write(
            f"Model Name: {model.__class__.__name__}\n"
        )


        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )


        f.write(
            f"Trainable Parameters: "
            f"{parameters:,}\n"
        )


        f.write(
            f"Trainable Parameters (Million): "
            f"{parameters/1e6:.3f} M\n"
        )


        # Model size
        temp_file = "temp_model.pth"

        torch.save(
            model.state_dict(),
            temp_file
        )

        size_MB = os.path.getsize(temp_file)/(1024**2)

        os.remove(temp_file)


        f.write(
            f"Model Size: {size_MB:.2f} MB\n"
        )


        f.write("\n")


        # -------------------------
        # Training Information
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Training Information\n")
        f.write("="*60 + "\n")


        if training_time:

            f.write(
                f"Training Time: "
                f"{training_time/3600:.3f} hours\n"
            )


        f.write("\n")


        # -------------------------
        # Hyperparameters
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Hyperparameters\n")
        f.write("="*60 + "\n")


        for key,value in hyperparameters.items():

            f.write(
                f"{key}: {value}\n"
            )


        f.write("\n")


        # -------------------------
        # Random Seeds
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Random Seeds\n")
        f.write("="*60 + "\n")


        if random_seeds:

            for seed in random_seeds:

                f.write(
                    f"{seed}\n"
                )


        f.write("\n")


        # -------------------------
        # Fold Results
        # -------------------------
        f.write("="*60 + "\n")
        f.write("Fold Results\n")
        f.write("="*60 + "\n")


        for i,fold in enumerate(fold_results):

            f.write(
                f"\nFold {i+1}\n"
            )

            for metric,value in fold.items():

                f.write(
                    f"{metric}: {value:.5f}\n"
                )


        f.write("\n")


        # -------------------------
        # Final Mean STD
        # -------------------------
        if mean_metrics:

            f.write("="*60 + "\n")
            f.write("Final Metrics (Mean ± STD)\n")
            f.write("="*60 + "\n")


            for metric in mean_metrics:

                if std_metrics:

                    f.write(
                        f"{metric}: "
                        f"{mean_metrics[metric]:.5f} ± "
                        f"{std_metrics[metric]:.5f}\n"
                    )

                else:

                    f.write(
                        f"{metric}: "
                        f"{mean_metrics[metric]:.5f}\n"
                    )


    print(f"Experiment information saved: {filepath}") 
#---------------------------------------------------------------------------
def HyperParametersToDict(hp):

    return {
        "percent": hp.percent,

        "batch_size": hp.batch_size,
        "num_epochs": hp.num_epochs,

        "lr_g": hp.lr_g,
        "lr_d": hp.lr_d,

        "lambda_L1": hp.lambda_L1,
        "lambda_L2": hp.lambda_L2,
        "lambda_g": hp.lambda_g,
        "lambda_ssim": hp.lambda_ssim,

        "beta1": hp.beta1,
        "beta2": hp.beta2
    }
#---------------------------------------------------------------------------
class HyperParameters():
    def __init__(self,batch_size=2,num_epochs=100,lambda_L1=0,lambda_L2=0,lambda_g=0,lambda_ssim=0):
        # Dataset
        self.percent = 0.81
        
        # Training
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Learning rates
        self.lr_g = 0.0002
        self.lr_d = 0.0001
        
        # Loss weights
        self.lambda_L1 = lambda_L1
        self.lambda_L2 = lambda_L2
        self.lambda_g = lambda_g
        self.lambda_ssim = lambda_ssim
        
        # Adam optimizer parameters
        self.beta1 = 0.5
        self.beta2 = 0.999
#---------------------------------------------------------------------------
class GAN:
    def __init__(self, generator, discriminator, parameters, train_loader,test_dataset,path):
        
        self.generator=generator
        self.discriminator=discriminator
        self.parameters=parameters
        self.optimizer_generator = torch.optim.Adam(generator.parameters(), lr=self.parameters.lr_g, betas=(self.parameters.beta1, self.parameters.beta2))
        self.optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=self.parameters.lr_d, betas=(self.parameters.beta1, self.parameters.beta2))

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generator, lr_lambda=lambda_lr)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator, lr_lambda=lambda_lr)
        self.device=GetDevice()
        self.loss_function = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()                      # L1 Loss
        self.l2_loss = nn.MSELoss()                     # L2 Loss
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)
       # self.train_dataset,self.test_dataset=Train_Test_Split_Dataset(dataset,self.parameters.percent)
        self.train_loader,self.test_loader=train_loader,test_dataset    #Train_Test_DataLoader(self.train_dataset, self.test_dataset,self.parameters.batch_size)
        self.resultspath = os.path.join(path,"Experiments_Results", generator.__class__.__name__,str(self.parameters.batch_size))
        os.makedirs(self.resultspath, exist_ok=True)

    def ssim_loss(self,fake, real):
            ssim_value = self.ssim_metric(fake, real)
            return 1 - ssim_value
    def Train(self,fold):
            best_g_loss = float("inf")
            best_d_loss = float("inf")

            losses_d = []
            losses_g = []
            for epoch in range(self.parameters.num_epochs):
                for n, (MRI,real_PET) in enumerate(self.train_loader):
            
                    real_PET = real_PET.to(self.device)
                    MRI = MRI.to(self.device)
                    # ---------------------
                    # Train Discriminator
                    # ---------------------

                    self.discriminator.zero_grad()

                    fake_PET = self.generator(MRI).detach()
                    # real pair  
                    d_real = self.discriminator(MRI, real_PET)
                    

                    real_labels = torch.ones_like(d_real).to(self.device)

                    loss_d_real = self.loss_function(d_real, real_labels)

                    # fake pair
                    d_fake = self.discriminator(MRI, fake_PET)
                

                    fake_labels = torch.zeros_like(d_fake).to(self.device)

                    loss_d_fake = self.loss_function(d_fake, fake_labels)

                    loss_discriminator = (loss_d_real + loss_d_fake) * 0.5

                    loss_discriminator.backward()
                    
                    self.optimizer_discriminator.step()

                    # ---------------------
                    # Train Generator
                    # ---------------------

                    self.generator.zero_grad()

                    fake_PET = self.generator(MRI)

                    output = self.discriminator(MRI, fake_PET)

                    real_labels = torch.ones_like(output).to(self.device)

                    loss_g_gan = self.loss_function(output, real_labels)*self.parameters.lambda_g

                    loss_g_l1 = self.l1_loss(fake_PET, real_PET) * self.parameters.lambda_L1

                    loss_g_l2 = self.l2_loss(fake_PET, real_PET) * self.parameters.lambda_L2

                    loss_ssim_l = self.ssim_loss(fake_PET, real_PET) * self.parameters.lambda_ssim 

                    loss_generator = loss_g_gan + loss_g_l1 + loss_g_l2 + loss_ssim_l   # 

                    loss_generator.backward()

                    self.optimizer_generator.step()

                    # 👇 BEST MODEL SAVE HERE
                    if loss_generator.item() < best_g_loss:
                        best_g_loss = loss_generator.item()
                        torch.save(self.generator.state_dict(), self.resultspath+rf"/{self.generator.__class__.__name__}_best_generator.pth")

                    if loss_discriminator.item() < best_d_loss:
                        best_d_loss = loss_discriminator.item()
                        torch.save(self.discriminator.state_dict(), self.resultspath+rf"/{self.generator.__class__.__name__}_best_discriminator.pth")

                    #print(f"Pair Image: {n+1}")

                # update learning rate end of each epoch
                self.scheduler_G.step()
                self.scheduler_D.step()

                losses_d.append(loss_discriminator.item())
                losses_g.append(loss_generator.item())
                #print(f"Epoch: {epoch+1}")

                if (epoch + 1) % 100 == 0:  # Each Two Epoch
                    print(f"Epoch:  {epoch+1}")
                    
                    print(f"LR G:   {self.scheduler_G.get_last_lr()[0]:.6f}")
                    print(f"LR D:   {self.scheduler_D.get_last_lr()[0]:.6f}")
                    print(f"Loss G: {loss_generator}")
                    print(f"Loss D: {loss_discriminator}")
                
                   
                    # Show_Loss(losses_g,losses_d)
            #self.SaveImage(self.generator,epoch+1,self.train_dataset,self.device)   
            Show(self.generator,epoch+1,self.test_loader,self.device,self.resultspath)     
            self.SaveLoss(losses_g,losses_d,self.resultspath,fold)
            torch.save(self.generator.state_dict(), self.resultspath+rf"/{self.generator.__class__.__name__}_last_generator.pth")
            torch.save(self.discriminator.state_dict(), self.resultspath+rf"/{self.generator.__class__.__name__}_last_discriminator.pth")
    def CalculateExperimentsResults(self,loader):
        self.generator.eval()
        fid,lpips=Evalution(self.generator,loader, self.device,self.resultspath).calculate()
       

        all_metrics = {
            "MSE": [],
            "MAE": [],
            "PSNR": [],
            "SSIM": [],
            "MSSSIM": []
        }

        with torch.no_grad():
            for mri,real in loader:
                for i in range(mri.size(0)):   #for each batch 
                    m = mri[i:i+1].to(self.device)
                    r = real[i:i+1].to(self.device)
                    f = self.generator(m)
                    # اگر grayscale → 3 کاناله (برای consistency)
                    if r.shape[1] == 1:
                        r = r.repeat(1, 3, 1, 1)
                        f = f.repeat(1, 3, 1, 1)
                    metrics = Metrics(r, f).calculate()
                    for key in all_metrics:
                        all_metrics[key].append(metrics[key])
        avg_metrics = {
            k: np.mean([x.detach().cpu().item() if torch.is_tensor(x) else x for x in v])
            for k, v in all_metrics.items()
        }
        avg_metrics["FID"] = float(fid)
        avg_metrics["LPIPS"] = float(lpips)
        # for key, value in avg_metrics.items():
        #     print(f"{key}: {value:.4f}")
        return avg_metrics

    def CalculateAndSaveExperimentsResults(self):
        avg_metrics=self.CalculateExperimentsResults(self.train_loader)
        self.SaveResults(avg_metrics,"Train")
        avg_metrics=self.CalculateExperimentsResults(self.train_loader)
        self.SaveResults(avg_metrics,"Test")
    def SaveResults(self,avg_metrics,std_metrics,str):
        os.makedirs(self.resultspath, exist_ok=True)
        save_file = os.path.join(self.resultspath,f"{self.generator.__class__.__name__}_{str}_evaluation_results.txt")
        with open(save_file, "w") as f:
            for key,value in avg_metrics.items():
                  f.write(f"{key}: {value:.4f} ± {std_metrics[key]:.4f}\n")
        # print(f"Metrics saved to: {save_file}")
    def SaveLoss(self,losses_g,losses_d,save_path,fold):
        plt.figure(figsize=(8,5))
        plt.plot(losses_g, label="Generator Loss", color="red")
        plt.plot(losses_d, label="Discriminator Loss", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("GAN Training Loss")
        plt.legend()
        plt.grid(True)

        # Save figure
        file_path = os.path.join(save_path,f"{self.generator.__class__.__name__}_Fold{fold}_LossFunction.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()
#---------------------------------------------------------------------------
class Metrics():
    def __init__(self,real,fake):
        self.real = real
        self.fake=fake
    def calculate(self):
        # print("REAL image:")
        # print("min:", self.real.min().item())
        # print("max:", self.real.max().item())

        # print("\nFAKE image:")
        # print("min:", self.fake.min().item())
        # print("max:", self.fake.max().item())
    
        metrics = {}

        # MSE
        metrics['MSE'] =self.calculateMSE()

        # MAE
        metrics['MAE'] = self.calculateMAE()

        # PSNR (range [-1,1] → MAX_I = 2)
        mse = metrics['MSE']
        metrics['PSNR'] = self.calculatePSNR(mse)
        
        metrics["SSIM"] = self.calculateSSIM()
        metrics["MSSSIM"] = self.calculateMS_SSIM()
        return metrics
    #----------------------------------------------------------------
    def calculateMSE(self):
        MSE = F.mse_loss(self.fake, self.real).item()
        return MSE
    def calculateMAE(self):
        MAE = F.l1_loss(self.fake, self.real).item()
        return MAE
    def calculatePSNR(self,mse):
        PSNR =20 * torch.log10(2.0 / torch.sqrt(torch.tensor(mse))).item()
        return PSNR
    def calculateSSIM(self):
        real = (self.real + 1) / 2
        fake = (self.fake + 1) / 2

        SSIM = ssim(real, fake, data_range=1.0)
        return SSIM
    def calculateMS_SSIM(self):
        real = (self.real + 1) / 2
        fake = (self.fake + 1) / 2

        real = F.interpolate(real, size=(256, 256), mode="bilinear", align_corners=False)
        fake = F.interpolate(fake, size=(256, 256), mode="bilinear", align_corners=False)

        MS_SSIM = multi_scale_ssim(real, fake, data_range=1.0)
        return MS_SSIM
#---------------------------------------------------------------------------
class Evalution():
    def __init__(self,generator,dataset,device,resultspath):
        self.generator = generator
        self.dataset=dataset
        self.device=device
        self.resultspath=resultspath
        self.real_path = os.path.join(resultspath, "Real")
        self.fake_path = os.path.join(resultspath, "Fake")
    def calculate(self):
        self.SaveImages()
        fid=self.calculateFID()
        #print(fid)
        lpips=self.calculateLPIPS()
        #print(lpips)
        return fid,lpips           
    def SaveImages(self):
        if os.path.exists(self.real_path):
            shutil.rmtree(self.real_path)

        if os.path.exists(self.fake_path):
            shutil.rmtree(self.fake_path)
        os.makedirs(self.real_path, exist_ok=True)
        os.makedirs(self.fake_path, exist_ok=True)
        self.generator.eval()

        index = 0

        with torch.no_grad():
            for MRI, Real_PET in self.dataset:
                MRI = MRI.to(self.device)
                Real_PET = Real_PET.to(self.device)
                Fake_PET = self.generator(MRI)
                batch_size = MRI.size(0)
                for b in range(batch_size):
                    # if Real_PET.shape[0] == 1:
                    #     Real_PET = Real_PET.repeat(3, 1, 1)fake.numpy()

                    # if Fake_PET.shape[0] == 1:
                    #     Fake_PET = Fake_PET.repeat(3, 1, 1)
                    # save_image(Real_PET[b],os.path.join(self.real_path, f"real_{index}.png"),normalize=True)
                    # save_image(Fake_PET[b],os.path.join(self.fake_path, f"fake_{index}.png"),normalize=True)
                    save_image(Real_PET[b].detach().cpu(),os.path.join(self.real_path, f"real_{index}.png"),normalize=True)
                    save_image(Fake_PET[b].detach().cpu(),os.path.join(self.fake_path, f"fake_{index}.png"),normalize=True)

                    index += 1                    
    def getImages(self):
        transform = transforms.ToTensor()

        real_images = []
        fake_images = []

        real_files = sorted(os.listdir(self.real_path))
        fake_files = sorted(os.listdir(self.fake_path))

        for index, (real_file, fake_file) in enumerate(zip(real_files, fake_files)):
            real = Image.open(os.path.join(self.real_path, real_file)).convert("L")
            fake = Image.open(os.path.join(self.fake_path, fake_file)).convert("L")

            real_images.append(transform(real))
            fake_images.append(transform(fake))
        return real_images,fake_images
    def calculateLPIPS(self):
       
        real_images, fake_images = self.getImages()
        with contextlib.redirect_stdout(io.StringIO()):
            lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        lpips_fn.eval()

        lpips_values = []

        with torch.no_grad():
            for real, fake in zip(real_images, fake_images):

                # [1,H,W] -> [1,1,H,W]
                real = real.unsqueeze(0)
                fake = fake.unsqueeze(0)

                # grayscale -> RGB [1,3,H,W]
                real = real.repeat(1, 3, 1, 1)
                fake = fake.repeat(1, 3, 1, 1)

                # move to device
                real = real.to(self.device)
                fake = fake.to(self.device)

                # [0,1] -> [-1,1]
                real = real * 2 - 1
                fake = fake * 2 - 1
               
                value = lpips_fn(real, fake)

                # GPU -> CPU before any numpy conversion
                value = value.detach().cpu()

                lpips_values.append(value.item())
       
        return sum(lpips_values) / len(lpips_values)
    # def calculateLPIPS(self):
    #     real_images, fake_images=self.getImages()
    #     #print("max:", real_images[0].max().item(),"min:", real_images[0].min().item())
    #     #print("max:", fake_images[0].max().item(),"min:", fake_images[0].min().item())

    #     lpips_fn = lpips.LPIPS(net='alex').to(self.device)  # or 'vgg'
    #     lpips_values = []

    #     lpips_fn.eval()

    #     with torch.no_grad():
    #         for real, fake in zip(real_images, fake_images):
    #             real = real.unsqueeze(0).to(self.device)
    #             fake = fake.unsqueeze(0).to(self.device)
    #             # 🔥 convert [0,1] → [-1,1]
    #             real = real * 2 - 1
    #             fake = fake * 2 - 1
    #             #print("max:", real.max().item(),"min:", real.min().item())
    #             #print("max:", fake.max().item(),"min:", fake.min().item())
    #             value = lpips_fn(real, fake)
    #             lpips_values.append(value.item())

    #     return sum(lpips_values) / len(lpips_values)
    def calculateFID(self):
        metrics = calculate_metrics(
            input1=self.real_path,
            input2=self.fake_path,
            fid=True,                                                              
            cuda=(self.device.type == "cuda"),
            isc=False,
            kid=False,
            verbose=False
        )

        return metrics["frechet_inception_distance"]
#---------------------------------------------------------------------------