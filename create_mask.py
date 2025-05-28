import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from mask_models import ViTExtractor
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import numpy as np
from utils.loss_utils import ssim

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomSegmentationDataset(Dataset):
    def __init__(self, original_dir, mask_dir, rendered_dir, transform=None, mask_transform=None, original_transform=None):
        self.original_dir = original_dir
        self.rendered_dir = rendered_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.original_transform = original_transform
        
        # Get all original images and corresponding masks
        self.original_images = sorted([f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))])
        self.rendered_images = sorted([f for f in os.listdir(rendered_dir) if os.path.isfile(os.path.join(rendered_dir, f))])

    def __len__(self):
        return len(self.original_images)  # Assuming the number of masks corresponds to the dataset length
    # return shape of the image
    def get_shape(self, idx):
        image_name = self.original_images[idx]
        original_path = os.path.join(self.original_dir, image_name)
        original_image = Image.open(original_path).convert('RGB')
        return original_image.size
    def __getitem__(self, idx):
        image_name = self.original_images[idx]
         
        # Load original image
        original_path = os.path.join(self.original_dir, image_name)
        original_image = Image.open(original_path).convert('RGB')

        # print(original_image.size)
        
        # Load corresponding rendered image
        rendered_path = os.path.join(self.rendered_dir, image_name)
        rendered_image = Image.open(rendered_path).convert('RGB')
        
        if self.transform:
            original_image = self.original_transform(original_image)
            rendered_image = self.transform(rendered_image)
        
        return original_image, rendered_image, image_name


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training Change Mask Segmentation Model')
    parser.add_argument('--load_size', default=518, type=int, help='load size of the input image.')
    parser.add_argument('--model_type', default='v2', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--input_folder', type=str, help="folder to load the images.", required=True)
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for training")
    parser.add_argument('--t', default=0.5, type=float, help="threshold value to make the mask")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)

    ref_dir = args.input_folder + "/renders/reference"
    query_dir = args.input_folder + "/renders/query"
    output_dir = args.input_folder + "/renders/noisy_masks"


    os.makedirs(output_dir, exist_ok=True)

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)),
        transforms.ToTensor()
    ])

    transform_original = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)),
        transforms.ToTensor(),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = CustomSegmentationDataset(original_dir=ref_dir, mask_dir=query_dir, rendered_dir=query_dir, transform=transform, mask_transform=mask_transform, original_transform=transform_original)
    # Set up data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Set up model
    extractor = ViTExtractor(model_type=args.model_type)
    h, w = dataset.get_shape(0)

    # make values < 0.5 to 0 and keep the rest
    t = args.t
    print(f"Threshold value: {t}")

    for batch in dataloader:
        original, rendered, image_name = batch

        ssim_map = ssim(original, rendered, map=True).mean(dim=0)        

        original = original.to(device)
        rendered = rendered.to(device)

        with torch.no_grad():
            original_features = extractor.extract_descriptors(original).squeeze(1)
            rendered_features = extractor.extract_descriptors(rendered).squeeze(1)

        # calculate the L1 distance between the features
        diff = torch.abs(original_features - rendered_features).sum(dim=-1)

        diff = diff.reshape(-1, 37, 37)

        # interpolate the diff to the original image size
        diff = nn.functional.interpolate(diff.unsqueeze(1), size=(w, h), mode='bicubic', align_corners=False).squeeze(1)
        ssim_map = nn.functional.interpolate(ssim_map.unsqueeze(1), size=(w, h), mode='bicubic', align_corners=False).squeeze(1)

        diff = (diff - diff.min()) / (diff.max() - diff.min())

        mask = (diff > args.t).float()

        diff = diff * mask

        ssim_map = (ssim_map < 0.5).float()

        intersect = diff.cpu() * ssim_map


        # save the diff as mask
        for i in range(len(image_name)):
            plt.imsave(os.path.join(args.input_folder, 'renders', 'noisy_masks' ,image_name[i].split('.')[0] + ".jpg"), intersect[i].cpu().numpy(), cmap='gray')
        