import torch
import torch.nn as nn
from collections import OrderedDict
from stylegan_layers import  G_mapping, G_synthesis
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class MappingNetworkGanInversionLoss:
    def __init__(self, net_type = "alex", device = "cpu"):
        """
        Loss function for training z and theta for GAN inversion task.

        loss := SquaredEuclideanDistance(y_hat, y) + LearnedPerceptualImagePatchSimilarity(y_hat, y)
        
        Args:
            lpips_type (str): type of neural net to use in the lpips loss
        """
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)

    def __call__(self, y_hat, y):
        """
        Args:
            y_hat (torch.tensor): predicted image with shape (N, C, W, H)
            y (torch.tensor): true image with shape (N, C, W, H)
        
        """
        assert y_hat.shape[1:] == y.shape[1:], \
            f"Tensor dims do not match:\n - y_hat {y_hat.shape}\n - y {y.shape}"

        # ||y_hat - y||^2_2
        squared_euclidean_norm = (y_hat - y).pow(2).mean()

        # squeeze pixel values into range [-1, 1]. Current method is to clamp
        y_hat = y_hat.clamp(-1, 1)
        lpips = self.lpips_loss(y_hat, y)

        return squared_euclidean_norm + lpips  


def init_gan(weights_path = 'weights/stylegan-weights.pt', device = None): 
    "Returns StyleGAN model initialized with pretrained weights"
    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=1024))    
        ]))
    if device is None:
        device = "mps" if getattr(torch,'has_mps',False
            ) else "cuda" if torch.cuda.is_available() else "cpu"
    #Load the pre-trained model
    g_all.load_state_dict(torch.load(weights_path, map_location=device))
    g_all.eval()
    g_all.to(device)
    return g_all


class MappingNetwork(nn.Module):
    def __init__(self, dim=512, hid=1024, output_act=None):
        """
        `MappingNetwork` is a three-layered MLP used for mapping the 
        Z-space to the X-Space (See Liu et al. 2022). Uses batchnorm after each dense connection.

        Args: 
            dim (int): input and output dimension. Represents dimension of the Z-space and X-space.
            hid (int): hidden size of the MLP. 1024 in the paper.
            output_act (torch.nn.functional): output activation function.

        NOTE: The output activation function is not specified in Liu et al. I am using batchnorm
        because the latent vector `x` is initialized from a distribution of reals with a mean of 
        zero and a standard deviation of 1. 
        """
        super().__init__()
        self.input_size = dim
        self.hidden_size  = hid
        self.fc1 = nn.Linear(dim, hid)
        self.bn1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.bn2 = nn.LayerNorm(hid)
        self.fc3 = nn.Linear(hid, dim)
        self.bn3 = nn.LayerNorm(dim)
        self.lrelu = nn.LeakyReLU(negative_slope=.2) 
        self.output_act = output_act

      #  self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.fc1(x)))
        x = self.lrelu(self.bn2(self.fc2(x)))
        out = self.bn3(self.fc3(x))
        out = self.output_act(out) if self.output_act else out
        return out

    def initialize_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


def load_mapping_network(path, device="cpu"):
    """
    Loads trained weights of mapping network theta.

    Args: 
        path (str): path to trained mapping network weights
        device (str or torch.device): device to put model on

    Returns: Model instantiated with trained weights 
    """
    if isinstance(device, str):
        device = torch.device(device)
    model = MappingNetwork()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model