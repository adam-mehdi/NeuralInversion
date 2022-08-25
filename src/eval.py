import torch
from tqdm import tqdm
from utils import control_gradient_flow, save_images, load_image
from models import init_gan, MappingNetworkGanInversionLoss, load_mapping_network


# ===== MODIFY `Configuration` TO CUSTOMIZE RUN =====
class Configuration:
    def __init__(self):
        """
        Configuration that specifies hyperparameters for inference.
        Modify attribute values yourself. 

        Attributes: 
            n (int): number of batches of images to generate when predicting image from latent vector 
            bs (int): batch size N, they managed N = 256 with 4-gpu setup
            dim (int): latent dim of Z and X (512). The StyleGan requires 512.
            lr_z (float): learning rate of input latent (.1)
            n_opt_steps (int): number of optimization steps on latent vector per sample T (20)
            clip_value (int): maximum value to which to clip gradients in optimizing theta      
            device (str): device to put model and tensors on 
                Note that mps is buggy due to official Python semaphore error, fixed in cpython 3.12.0alpha.
            n_images (int): number of images to train on
            criterion: loss function. 
                MappingNetworkGanInversionLoss := MeanEuclideanDistance + LearnedPerceptualImagePatchSimilarity
            weights_path_gan (str): path to pretrained StyleGAN weights
            weights_path_mapping_network (str): path to pretrained mapping network weights
            optimizer_z: optimizer for latent vector z (Adam)

        NOTE: The ending values are the ones used in the 'Learning Loss
              Landscapes' paper.
        """
        self.n = 1
        self.bs = 2                             
        self.dim = 512                          
        self.lr_z = .1 * 10**-2                 
        self.n_opt_steps = 20                   
        self.clip_value = 10       
        self.device = torch.device("cpu")#"mps" if getattr(torch,'has_mps', False) \
            #else "cuda" if torch.cuda.is_available() else "cpu")
        self.n_images = 1
        self.weights_path_gan = 'weights/stylegan-weights.pt'
        self.weights_path_mapping_network = "weights/trained_theta_weights.pth"
        self.criterion = MappingNetworkGanInversionLoss(net_type="vgg", device=self.device)

    
    def init_optimizers(self, latent_vector_z):
        self.optimizer_z = torch.optim.Adam([latent_vector_z], lr=self.lr_z)



def image2vectors(gan, map_net, image, cfg):
    """
    Returns latent vectors corresponding with `image`.

    Args:
        gan (nn.Module): frozen gan model
        map_net (nn.Module): trained mapping network
        image (torch.Tensor): image
        cfg (Configuration)
    """
    z = torch.zeros(cfg.bs, cfg.dim, requires_grad=True, device=cfg.device)
    loss_z = []
    
    cfg.init_optimizers(z)

    if len(image) == 3: 
        image = image.unsqueeze(0)
    assert image.shape == torch.Size([1, 3, 1024, 1024]), \
        f"Image dimensions must be [1, 3, 1024, 1024]. Currently, they are {image.shape}."

    map_net = control_gradient_flow(map_net, enable_grad=False)
    map_net.eval()
    
    for t in tqdm(range(cfg.n_opt_steps), total=cfg.n_opt_steps):
        y_hat = gan(map_net(z))
        assert not y_hat.isnan().any(), print("y_hat is nan when optimizing z", y_hat, t)

        loss = cfg.criterion(y_hat, image) 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(z, cfg.clip_value)
        cfg.optimizer_z.step()
        cfg.optimizer_z.zero_grad()

        print(f"STEP {t}\t LOSS {loss.item()}")
        loss_z.append((t, loss.item()))
    
    return z
        

def vectors2images(gan, map_net, z, cfg):
    """
    Returns a list of n 

    Args:
        gan (nn.Module): frozen gan model
        map_net (nn.Module): trained mapping network
        z (torch.Tensor): image
        n (int): generate
        cfg (Configuration)
    
    Returns: list of torch.tensors with shape (N, C, H, W).
    """
    if len(z.shape) == 1: 
        z = z.unsqueeze()

    assert z.shape == torch.Size([cfg.bs, cfg.dim]), \
        f"latent vector z must have shape (bs, dim) in Configuration. Found {z.shape}."

    images = []

    for i in tqdm(range(cfg.n), total=cfg.n):
        imgs = gan(map_net(z))
        images.append(imgs)

    return images


if __name__ == "__main__":
    cfg = Configuration()
    image = load_image(device = cfg.device)

    gan = init_gan().to(cfg.device)
    map_net = load_mapping_network(cfg.weights_path, cfg.device)

    z = image2vectors(gan, map_net, image, cfg)
    image_preds = vectors2images(gan, map_net, z, cfg)

    save_images(image_preds)

    print("COMPLETED.")

    