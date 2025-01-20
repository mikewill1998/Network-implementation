from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as ds
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchinfo
from torchinfo import summary


cudnn.benchmark = True

manualSeed = 42
print("Seed: ", manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)


# dataset = ds.ImageFolder(root='./MNIST_data',
#                            transform=transforms.Compose([
#                            transforms.Resize(28),
#                            transforms.RandomRotation(60),
#                            transforms.ToTensor(),
#                        ]))
dataset = ds.MNIST(root='./MNIST_data',
                   download=True,
                    transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.RandomRotation(60),
                    transforms.ToTensor()
                ]))

nc=1

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ngpu = 1
nz = 100
ngf = 64
ndf = 64

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

netG = Generator(ngpu).to(device)
netG.apply(init_weight)
netG.load_state_dict(torch.load('weights/netG.pth', map_location=device, weights_only=True))
print(netG)
summary(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(init_weight)
netD.load_state_dict(torch.load('weights/netD.pth', map_location=device, weights_only=True))
print(netD)
summary(netD)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.99))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

epochs = 10
generator_loss = []
discriminator_loss = []
for epoch in range(epochs):
    g_loss = 0
    d_loss = 0
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_on_cpu = data[0].to(device)
        batch_size = real_on_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)

        output = netD(real_on_cpu)
        errD_real = criterion(output, label)
        d_loss += errD_real.cpu().detach().numpy()
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        g_loss += errD_fake.cpu().detach().numpy()
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            fake = netG(fixed_noise)
        
    d_loss /= len(dataloader)
    g_loss /= len(dataloader) 
    discriminator_loss.append(d_loss)
    generator_loss.append(g_loss)


torch.save(netG.state_dict(), f"netG_{epoch+1}.pth")
torch.save(netD.state_dict(), f"netD_{epoch+1}.pth")

num_epoch = [e for e in range(epochs)]

plt.plot(num_epoch, discriminator_loss, label="discriminator")
plt.plot(num_epoch, generator_loss, label="generator")
plt.title("Loss")
plt.legend()
plt.show()