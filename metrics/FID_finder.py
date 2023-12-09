from PIL import Image
import numpy as np
import os
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (512, 512))

dataset_path = "Major_truth_FID"
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

dataset_path2 = "Riffusion_fake_FID"
image_paths2 = sorted([os.path.join(dataset_path2, x) for x in os.listdir(dataset_path2)])

fake_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths2]

real_images = torch.cat([preprocess_image(image) for image in real_images])
# print(real_images.shape)

fake_images = torch.cat([preprocess_image(image) for image in fake_images])
# print(fake_images.shape)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
# FID: 177.7147216796875