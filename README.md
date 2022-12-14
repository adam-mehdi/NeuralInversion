# `NeuralInversion`
<center> <img src="neural-inversion-logo.png" width="200" height="150"> </center>

`NeuralInversion` approximates the inverse of neural networks. It currently implements inverse 
for StyleGAN, meaning it can take an image and find the feature vector in StyleGAN latent space with 
which it corresponds. 

## What it does

To invert neural networks, `NeuralInversion` uses a modified version of *optimizer-based inference* (OBI). OBI calculates the inverse mapping as follows. 
Given an image $y$, initialize a latent vector $x$ and pass it through the pretrained StyleGAN $F$ to compute 
the loss $L(F(x), y)$. $^1$ Take the partial derivatives of $L$ with respect to $x$ and minimize $L$ using gradient 
descent by optimizing $x$. Now pass the new $x$ into $F$ again, and repeat the optimization process for a number of times $T$ (usually $T=20$). $^2$ 
The problem with this OBI procedure is that the optimization is unstable: The StyleGan $F$ was designed to optimize its parameters, not its input.
Hence, Liu et al. provide a method to stabilize optimization by smoothing the loss landscape in "Landscape Learning for Neural Network Inversion".
They do this by training another neural network that predict the input vector $x \in X$ using a vector from another space $z \in Z$
with a smooth loss landscape. The mapping network $\theta$ is trained to minimize the loss $L(F(\theta(z_t)), y)$ where $t\in T$. $^3$ From this training,
$\theta$ learns patterns in the optimization trajectories of $X$ and can act to stabilize them, learning a loss landscape where gradient descent is efficient, 
and accelerating the inversion process.

## How to use
First, install the dependencies needed to use `NeuralInversion` as follows. 

```shell
conda create -n neuralinversion python
conda activate neuralinversion
conda install -c conda-forge numpy torchvision tqdm
pip install torchmetrics
```

Make sure to activate your conda environment and `cd` into the `NeuralInversion` directory. 
And before running any scripts, modify the `Configuration` class in the `train` or `eval` 
python files according to your run configuration.

To train from scratch, run train.py as follows.

```
python src/train.py
```

If you already have pretrained weights for the mapping network, run the following line.

```
python src/eval.py
```

The pretrained weights for the StyleGAN are available in this [Google Drive folder](https://drive.google.com/drive/folders/1Qn5RtRdOuhA3eLsBGppTNx9v4zLZFRru?usp=sharing). 
Weights for the mapping network are also available there, but note that they are not fully pretrained; they have been trained on around 10 images.

When the `eval.py` script is completed, you will see the output images in the `output` directory.

## Technical Notes
**[1]** The loss function used here is as follows.

$$L(\hat{y}, y) := \frac{1}{volume(y)} \cdot SquaredEuclideanDistance(\hat{y}, y) + LearnedPerceptualImagePatchSimilarity(\hat{y}, y)$$

Here, both the predictions and targets are images with shape `N, 3, 1024, 1024` where `N` is the batch size. 
* `SquaredEuclideanDistance` measures how far apart corresponding pixel values are. 
* `LearnedPerceptualImagePatchSimilarity` uses a neural network
such as AlexNet to classify the semantic similarity between two images. See my article 
[Image Similarity: Theory and Code](https://towardsdatascience.com/image-similarity-theory-and-code-2b7bcce96d0a) for the essentials of image similarity.  

---

**[2]** Mathematically, the OBI procedure iterates the following line.

$$x_{t, i} = x_{t-1, i} - \alpha\frac{\partial L}{\partial x_i}$$ 

---

**[3]** Precisely, this entails first collecting together optimization trajectories into an N-by-T matrix with
N latent vector trajectories where each trajectory comprises T latent vectors. Each of the T latent vectors in a trajectory
is defined as follows.

$$z_{t, i} = z_{t-1, i} + \alpha\frac{\partial L}{\partial x_i}$$ 

Then, iteratively sample a latent vector and minimize 

$$L(F(\theta(z_{i,t})), y_i)$$ 

by performing gradient descent on each weight of the mapping network

$$w_{t, i} = x_{t-1, i} + \alpha\frac{\partial L}{\partial w_i}$$

Repeat until convergence.

---

**[4]** The mapping network was only trained on 10 images. 
Liu et al. trained the mapping network on a 4-GPU cluster with 256 data samples per buffer, 
but I was only able to support around 4 samples per buffer on my local machine
due to memory constraints.

## Preliminary Results

Using the pretrained weights for the StyleGAN linked above in the Google Drive.

Take the following input image of the blonde supermodel.

<img src="data/1.jpg" width=250> 

If you use `NeuralInvert` to find a latent vector that corresponds with it, and then you
pass that latent vector back into the StyleGAN, you get the following image.

<img src="output/image-output0-version0.png" width=250> 

Make what you will of the correspondence between the two images (maybe you're thinking she looks like a chubby middle-school
version of the supermodel; imagine what `NeuralInversion` will make of us mere mortals!). Nevertheless, if you want better performance, train the mapping network for longer on a 
dataset of images. One good source of images is the [CelebAMask dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html#:~:text=CelebAMask%2DHQ%20is%20a%20large,facial%20attributes%20corresponding%20to%20CelebA) (the blonde supermodel is a sample of this dataset).

## Resources and References
```
@misc{https://doi.org/10.48550/arxiv.2206.09027,
  doi = {10.48550/ARXIV.2206.09027},
  url = {https://arxiv.org/abs/2206.09027},
  author = {Liu, Ruoshi and Mao, Chengzhi and Tendulkar, Purva and Wang, Hao and Vondrick, Carl},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Landscape Learning for Neural Network Inversion},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@inproceedings{CelebAMask-HQ,
  title = {MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author = {Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

```
Pretrained weights for the StyleGAN were obtained in an [Image2StyleGAN repository](https://github.com/zaidbhat1234/Image2StyleGAN).
