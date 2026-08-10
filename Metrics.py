import os
import torch
import lpips
import numpy as np
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import shutil
from piq import ssim, multi_scale_ssim
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
torch.manual_seed(111)  # Random Generator Se

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
class Evalution():
    def __init__(self,generator,dataset,device):
        self.generator = generator
        self.dataset=dataset
        self.device=device
        self.real_path=r"Real"
        self.fake_path=r"Fake"
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
                Fake_PET = self.generator(MRI)
                batch_size = MRI.size(0)
                for b in range(batch_size):
                    # if Real_PET.shape[0] == 1:
                    #     Real_PET = Real_PET.repeat(3, 1, 1)

                    # if Fake_PET.shape[0] == 1:
                    #     Fake_PET = Fake_PET.repeat(3, 1, 1)
                    save_image(Real_PET[b],os.path.join(self.real_path, f"real_{index}.png"),normalize=True)
                    save_image(Fake_PET[b],os.path.join(self.fake_path, f"fake_{index}.png"),normalize=True)
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
        real_images, fake_images=self.getImages()
        #print("max:", real_images[0].max().item(),"min:", real_images[0].min().item())
        #print("max:", fake_images[0].max().item(),"min:", fake_images[0].min().item())

        lpips_fn = lpips.LPIPS(net='alex').to(self.device)  # or 'vgg'
        lpips_values = []

        lpips_fn.eval()

        with torch.no_grad():
            for real, fake in zip(real_images, fake_images):
                real = real.unsqueeze(0).to(self.device)
                fake = fake.unsqueeze(0).to(self.device)
                # 🔥 convert [0,1] → [-1,1]
                real = real * 2 - 1
                fake = fake * 2 - 1
                #print("max:", real.max().item(),"min:", real.min().item())
                #print("max:", fake.max().item(),"min:", fake.min().item())
                value = lpips_fn(real, fake)
                lpips_values.append(value.item())

        return sum(lpips_values) / len(lpips_values)
    def calculateFID(self):
        metrics = calculate_metrics(
            input1=self.real_path,
            input2=self.fake_path,
            fid=True,                                                              
            cuda=False,
            isc=False,
            kid=False,
            verbose=False
        )

        return metrics["frechet_inception_distance"]
    
def evaluate_test_set(generator, test_loader, device):
    fid,lpips=Evalution(generator, test_loader, device).calculate()
    generator.eval()

    all_metrics = {
        "MSE": [],
        "MAE": [],
        "PSNR": [],
        "SSIM": [],
        "MSSSIM": []
    }

    with torch.no_grad():
        for mri,real in test_loader:
            for i in range(mri.size(0)):   #for each batch 
                m = mri[i:i+1]
                r = real[i:i+1]
                f = generator(m)
                # اگر grayscale → 3 کاناله (برای consistency)
                if r.shape[1] == 1:
                    r = r.repeat(1, 3, 1, 1)
                    f = f.repeat(1, 3, 1, 1)
                metrics = Metrics(r, f).calculate()

               # print(metrics)

                for key in all_metrics:
                    all_metrics[key].append(metrics[key])

    # میانگین کل دیتاست
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"FID: {fid:.4f}")
    print(f"LPIPS: {lpips:.4f}")
    return avg_metrics
