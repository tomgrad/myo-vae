import torch
from torch import nn
from torch.nn import functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super(Downsample, self).__init__()
        if with_conv:
            self.down = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.down(x)
        return x




class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResnetBlock1D, self).__init__()
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.nonlin = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.proj_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.nonlin(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.nonlin(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.in_channels != self.out_channels:
            residual = self.proj_conv(residual)

        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, latent_dim, f=8):
        super(Encoder, self).__init__()
        self.seq = nn.Sequential(
        nn.Conv1d(1, f, 1),
        ResnetBlock1D(f, 2*f),
        Downsample(2*f),
        ResnetBlock1D(2*f, 4*f),
        Downsample(4*f),
        ResnetBlock1D(4*f, 8*f),
        Downsample(8*f),
        ResnetBlock1D(8*f, 16*f),
        Downsample(16*f),
        ResnetBlock1D(16*f, 32*f),
        Downsample(32*f),
        nn.Flatten(),
        nn.Linear(32*f*8, 256),
        nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.seq(x)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, f):
        super(Decoder, self).__init__()
        self.seq = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 32*f*8),
        nn.ReLU(),
        nn.Unflatten(1, (32*f, 8)),
        ResnetBlock1D(32*f, 16*f),
        Upsample(16*f),
        ResnetBlock1D(16*f, 8*f),
        Upsample(8*f),
        ResnetBlock1D(8*f, 4*f),
        Upsample(4*f),
        ResnetBlock1D(4*f, 2*f),
        Upsample(2*f),
        ResnetBlock1D(2*f, f),
        Upsample(f),
        nn.Conv1d(f, 1, 1),
        nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


class VAE(nn.Module):
    def __init__(self, laten_dim, f):
        super(VAE, self).__init__()
        self.enc = Encoder(laten_dim, f)
        self.dec = Decoder(laten_dim, f)
        
    def forward(self, x):
        mu, logvar = self.enc(x)
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        z = mu+sigma*epsilon
        x = self.dec(z)
        return x, mu, logvar

if __name__ == '__main__':
    x = torch.randn(1, 32, 64)
    block = ResnetBlock1D(32, 64)
    y = block(x)
    print(y.size())