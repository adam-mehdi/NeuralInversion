import os
import torch
import os.path as op
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

def control_gradient_flow(block, enable_grad=True):
    """
    Freezes or unfreezes the model or tensor.
    Args:
        block: mapping network (theta) or torch tensor (z)
        enable_grad: whether to enable gradients on `block`
    """
    if not hasattr(block, "parameters"):
        block.requires_grad = enable_grad
    else:
        for param in block.parameters():
            param.requires_grad = enable_grad
    return block


def load_image(file_name = '0.jpg', data_dir = 'data', device = None, size = 1024):
    """
    Opens the image file in `data_dir/file_name`, putting it on `device` and resizing it to size `size`.
    Returns: torch.tensor
    """
    img_path = op.join(data_dir, file_name)

    with open(img_path,"rb") as f: 
        image=Image.open(f)
        image=image.convert("RGB")

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def get_images(n_images, data_dir = 'data', device = None, size = 1024, sort = False):
    """
    Retrieves `n_images` images from `data_dir`, resizes them to `size`, 
    and puts them on `device`.
    
    Args:
        n_images (int): number of images to retrieve
        data_dir (str): directory path from which to retrieve the images
        device (str): device to put images on. Inferred automatically if `None`.
        size (int): width and height to which to resize the images
        sort (bool): whether to sort images. Assumes file name structure `{int}.jpg`.

    Returns: List with `n_images` images
    """
    if device is None:
        device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"

    img_files = os.listdir(data_dir)
    if sort: 
        img_files.sort(key=lambda x: int(x.split(".")[0])) # optionally sort images

    img_files = img_files[:n_images]
    images = [load_image(fname, data_dir, device, size) for fname in img_files]
    return images


def generate_n_random_images(model, n: int, device = None):
    """Given a trained GAN `model`, saves `n` random outputs to the `outputs` dir."""
    if device is None:
        device = "mps" if getattr(torch,'has_mps',False
                ) else "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(range(n)):
        z = torch.randn(1,512, device = device)
        img = model(z)
        img = ((img +1.0)/2.0).clamp(0,1)
        save_image(img[0], f"output/random-output-{i}.png")


def save_images(images, id = ""):
    """
    Saves images to `output` folder.

    Args:
        images (List[List]): 2D list where the image at row i and column j
                             is the ith version of the jth image in the batch
        id (str): identifier to append to the image file name
    """
    for i in range(len(images)):
        for v in range(len(images[i])):
            img = images[i][v]
            img = ((img +1.0)/2.0).clamp(0,1)
            save_image(img, f"output/image-output{i}-version{v}{id}.png")
