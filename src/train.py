import os.path as op
from random import randint
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from pickle import dump
import torch.nn.functional as F
from models import init_gan, MappingNetwork, MappingNetworkGanInversionLoss
from utils import get_images, control_gradient_flow

# ===== MODIFY `Configuration` TO CUSTOMIZE RUN =====
class Configuration:
    def __init__(self):
        """
        Configuration that specifies hyperparameters for inference.
        Modify attribute values yourself. 

        Attributes: 
            bs (int): batch size N, they managed N = 256 with 4-gpu setup
            dim (int): latent dim of Z and X (512). The StyleGan requires 512.
            lr_z (float): learning rate of input latent (.1)
            lr_theta (float): learning rate of mapping net (.0001)
            n_buffers (int): number of experience replay buffers B (500)
            n_opt_steps (int): number of optimization steps on latent vector per sample T (20)
            wd (float): weight decay for mapping network
            clip_value (int): maximum value to which to clip gradients in optimizing theta      
            device (torch.device): device to put model and tensors on 
                Note that mps is buggy due to official Python semaphore error, fixed in cpython 3.12.0alpha.
            n_images (int): number of images to train on
            data_path (str): path to folder with image data
            criterion: loss function. 
                MappingNetworkGanInversionLoss := MeanEuclideanDistance + LearnedPerceptualImagePatchSimilarity
            optimizer_theta: optimizer for mapping network theta (AdamW)
            optimizer_z: optimizer for latent vector z (Adam)

        NOTE: The ending values are the ones used in the 'Learning Loss
              Landscapes' paper.
        """
        self.bs = 2                             # batch size N, they managed N = 256 with 4-gpu setup
        self.dim = 512                          # latent dim of Z and X (512)
        self.lr_z = .1 * 10**-2                 # learning rate of input latent 
        self.lr_theta = .0001 * 10**-2          # learning rate of mapping net (.0001)
        self.n_buffers = 10                     # number of buffers B (500)
        self.n_opt_steps = 20                   # number of optimization steps per sample T (20)
        self.wd = .1                            # weight decay for mapping network
        self.clip_value = 10                    # maximum value to which to clip gradients in optimizing theta
        self.device = torch.device("mps" if getattr(torch,'has_mps', False) \
            else "cuda" if torch.cuda.is_available() else "cpu")
        self.n_images = 5
        self.data_path = 'data'
        self.criterion = MappingNetworkGanInversionLoss(net_type="vgg", device=self.device)
    
    def init_optimizers(self, mapping_network, latent_vector_z):
        self.optimizer_theta = torch.optim.AdamW(mapping_network.parameters(), weight_decay=self.wd, lr=self.lr_theta)
        self.optimizer_z = torch.optim.Adam([latent_vector_z], lr=self.lr_z)

def log_loss(lossdict, image_num, buffer_num, opt_step, loss):
    lossdict["image_num"].append(image_num)
    lossdict["buffer_num"].append(buffer_num)
    lossdict["opt_step"].append(opt_step)
    lossdict["loss"].append(loss.item())

                

def train_mapping_network(gan, map_net, z, dataset, cfg):
    """
    Trains the mapping network theta using data.

    Args:
        gan (nn.Module): frozen gan model
        map_net (nn.Module): mapping network to train
        dataset (list): list of data samples to train the mapping network with
    """
    device = cfg.device
    loss_z = defaultdict(list)
    loss_theta = defaultdict(list)
    for image_num, y in enumerate(dataset):
        for b in tqdm(range(cfg.n_buffers)):
            print("BUFFER NUMBER", b)
            # initialize experience replay buffer, shape (B, N, D)
            buffer = torch.zeros(cfg.n_opt_steps, cfg.bs, cfg.dim, device=device)
            # disable gradients for mapping network
            map_net = control_gradient_flow(map_net, enable_grad=False)

            # populate experience replay buffer with T vectors of z, 
            # where z_t is the gradient update of z_t-1
            
            for t in range(cfg.n_opt_steps):
                print("theta_opt_step:", t)

                y_hat = gan(map_net(z))
                assert not y_hat.isnan().any(), print("y_hat is nan when training z", y_hat, t)

                loss = cfg.criterion(y_hat, y) 
                loss.backward()

                torch.nn.utils.clip_grad_norm_(z, cfg.clip_value)
                cfg.optimizer_z.step()
                cfg.optimizer_z.zero_grad()

                buffer[t] = z.clone().detach()
                log_loss(loss_z, image_num, b, t, loss)
                

            # Enable gradient for theta and disable for z
            map_net = control_gradient_flow(map_net, enable_grad=True)
            buffer = control_gradient_flow(buffer, enable_grad=False)

            if buffer.isnan().any():
                print(f"Buffer {b} has nan values. This indicates unstable training; lower the learning rate.")
                buffer = buffer.nan_to_num() 

            # Optimize mapping network using experience replay buffer
            for j in range(cfg.n_opt_steps):
                print("theta_opt_step:", j)
                t = randint(0, cfg.n_opt_steps-1)
               # i = randint(0, cfg.bs-1)

                z = buffer[t] #, i].unsqueeze(0)
                y_hat = gan(map_net(z))
                assert not y_hat.isnan().any(), print("y_hat is nan when training theta", y_hat, j)
                
                loss = cfg.criterion(y_hat, y) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(map_net.parameters(), cfg.clip_value)
                cfg.optimizer_theta.step()
                cfg.optimizer_theta.zero_grad() 

                log_loss(loss_theta, image_num, b, j, loss)
    
    with open('output/loss_theta.pkl', 'wb') as f: 
        dump(loss_theta, f)
    with open('output/loss_theta.pkl', 'wb') as f: 
        dump(loss_theta, f)

    np.save('output/loss_z.npy', loss_z) 
    np.save('output/loss_theta.npy', loss_theta) 
    return map_net
        



if __name__ == "__main__":
    cfg = Configuration()
    images = get_images(cfg.data_path, cfg.n_images)

    gan = init_gan().to(cfg.device)
    map_net = MappingNetwork().to(cfg.device)
    z = torch.zeros(cfg.bs, cfg.dim, requires_grad=True, device=cfg.device)
    cfg.init_optimizers(map_net, z)

    trained_theta = train_mapping_network(gan, map_net, z, images, cfg)
    torch.save(trained_theta.state_dict(), 'output/trained_theta_weights.pth')

    # Load
    #read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
    print("COMPLETED.")

    