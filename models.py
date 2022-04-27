import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import MSELoss

class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.InstanceNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class Decoder(nn.Module):
    def __init__(self, size):
        super(Decoder, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net


class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net

class DiscVAE(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((1024 - self.sep * 2) * self.size * self.size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (1024 - self.sep * 2) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net
    
class VAE1(nn.Module):
  def __init__(self, sep, size):
    super(VAE1, self).__init__()
    self.sep = sep
    self.size = size

    self.full = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
        nn.InstanceNorm2d(512 - self.sep),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d((512 - self.sep), (1024 - self.sep * 2), 4, 2, 1),
        nn.InstanceNorm2d(1024 - self.sep * 2),
        nn.LeakyReLU(0.2, inplace=True),
    )

  def forward(self, net):
    net = self.full(net)
    net = net.view(-1, (1024 - self.sep * 2) * self.size * self.size)
    return net
    
class VAE2(nn.Module):
  def __init__(self, sep, size):
    super(VAE2, self).__init__()
    self.sep = sep
    self.size = size

    self.full = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, self.sep * 2, 4, 2, 1),
        nn.InstanceNorm2d(self.sep * 2),
        nn.LeakyReLU(0.2),
    )

  def forward(self, net):
    net = self.full(net)
    net = net.view(-1, self.sep * 2 * self.size * self.size)
    return net

class DecoderVAE(nn.Module):
  def __init__(self, size):
    super(DecoderVAE, self).__init__()
    self.size = size

    self.main = nn.Sequential(
        nn.ConvTranspose2d(512, 512, 4, 2, 1),
        nn.InstanceNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.InstanceNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 3, 4, 2, 1),
        nn.Tanh()
    )

  def forward(self, net):
    z = net.view(-1, 1024, self.size, self.size)
    half = z.shape[1] // 2
    z_mean = z[:, :half, :, :]
    z_std = z[:, half:, :, :]
    
    z_std = torch.exp(z_std)
    z = Normal(z_mean, z_std).sample()
    z = z.mul(z_std).add_(z_mean)

    net = self.main(z)
    return (z_mean, z_std), net

class NegativeELBOLoss:
  def __init__(self, beta=1):
    self.beta = beta
  
  def __call__(self, outs, x):
    (z_mean, z_std), x_hat = outs
    q = Normal(z_mean, z_std)
    p = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
    mse = MSELoss()
    return mse(x_hat, x) + self.beta * kl_divergence(q, p).mean()
