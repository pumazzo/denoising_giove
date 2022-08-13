import torch
import torch.nn as nn

class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
            
            
class UpNet(nn.Module):

    def __init__(self, in_channels = 2, out_channel=2, layer=12, kernel_size=[3,7],padding=(1,3)):
        super(UpNet, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding), 
                BatchRenorm2d(32), 
                nn.ReLU()]

        for i in range(layer):
            layers.append(nn.Conv2d(32, 32, kernel_size, 1,padding=padding))
            layers.append(BatchRenorm2d(32))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, out_channel, 3, 1,1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class DownNet(nn.Module):

    def __init__(self, kernel_size=[3,7]):
        super(DownNet, self).__init__()
        layers = [nn.Conv2d(in_channels=2, out_channels=32, kernel_size=kernel_size, stride=1, padding=(1,3)), 
                BatchRenorm2d(32), 
                nn.ReLU()]

        for i in range(3):
            layers.append(nn.Conv2d(32, 32, [3,3], 1, padding=(1,2), dilation=(1,2)))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, 3, 1, 1))
        layers.append(BatchRenorm2d(32))
        layers.append(nn.ReLU())
        for i in range(2):
            layers.append(nn.Conv2d(32, 32, [3,3], 1, padding=(2,2), dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, 3, 1, 1))
        layers.append(BatchRenorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 2, [3,7], 1, padding=(1,3)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out
    
class freq2Image(nn.Module):

    def __init__(self):
        super(freq2Image, self).__init__()
         
        self.subs= torch.nn.AvgPool2d([1,2],stride=[1,2])

    def forward(self, x):
        
        
        subs = self.subs(x)
        z = torch.complex(subs[:,0,:,:], subs[:,1,:,:])
        z = torch.unsqueeze(z,dim=1)
        im = torch.fft.fftshift( torch.fft.ifft2( torch.fft.ifftshift(z,dim=[-2,-1] ),norm="ortho" ),dim=[-2,-1]).abs()
        
        return  im
    
    

class BRDNet(nn.Module):

    def __init__(self):
        super(BRDNet, self).__init__()
        self.upnet = UpNet(layer=8)
        self.dwnet = DownNet()
        self.conv = nn.Conv2d(4, 2, [3,7], 1, padding=(1,3))
        
        #self.magnet=UpNet(in_channels = 3, out_channel=1, layer=1, kernel_size=3,padding=(1,1))
        self.last = nn.Conv2d(3, 1, 3, 1,1)
        
        self.subs= torch.nn.AvgPool2d([1,2],stride=[1,2])
        
        self.im1 = freq2Image()
        self.im2 = freq2Image()
        self.im_tot = freq2Image()

    def forward(self, x):
        #import pdb;pdb.set_trace()
        out1 = self.upnet(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        
        im1 = self.im1(out1)
        im2 = self.im1(out2)
        
        im = self.im_tot(out)
        
        im_u = torch.cat((im, im1, im2), 1)
        
        #im = im - self.magnet(im_u)
        im = self.last(im_u)
        
        return out, im
    
    
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17,features = 64):
        super(DnCNN, self).__init__()
        kernel_size = [3,7]
        padding = [1,3]
        #features = features
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        
        self.subs= torch.nn.AvgPool2d([1,2],stride=[1,2],divisor_override=2)
        
    def forward(self, x):
        out = x - self.dncnn(x)
        subs = self.subs(out)
        
        z = torch.complex(subs[:,0,:,:], subs[:,1,:,:])
        #print(z.dtype)
        z = torch.unsqueeze(z,dim=1)
        #print(z.dtype)
        im = torch.fft.fftshift( torch.fft.ifft2( torch.fft.ifftshift(z,dim=[-2,-1] ),norm="ortho" ),dim=[-2,-1]).abs()

        
        return out, im