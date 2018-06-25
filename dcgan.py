import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from PIL import Image
import os


class DCGAN:
    def __init__(self, generator,
                 discriminator,
                 device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    def fit(self, data_loader,
            nb_epoch=100,
            lr_d=1e-3,
            lr_g=1e-3,
            save_steps=10,
            logdir='logs'):
        os.makedirs(logdir, exist_ok=True)

        opt_d = optim.Adam(self.discriminator.parameters(), lr_d,
                           betas=(0.5, 0.999))
        opt_g = optim.Adam(self.generator.parameters(), lr_g,
                           betas=(0.5, 0.999))

        criterion = nn.BCELoss()
        for epoch in range(1, nb_epoch):
            print('\nepoch %d / %d' % (epoch, nb_epoch))
            start = time.time()
            for iter_, x in enumerate(data_loader):
                x = x[0]
                # update discriminator
                self.discriminator.zero_grad()
                x_real = x.to(self.device)
                bs = x_real.shape[0]
                t_real = torch.full((bs, ), 1, device=self.device)
                d_x_real = self.discriminator(x_real)
                loss_d_real = criterion(d_x_real, t_real)
                loss_d_real.backward()

                z = torch.randn((bs, self.generator.input_dim))
                x_fake = self.generator(z)
                t_fake = torch.full((bs, ), 0, device=self.device)
                d_x_fake = self.discriminator(x_fake)
                loss_d_fake = criterion(d_x_fake, t_fake)
                loss_d_fake.backward()

                loss_d = loss_d_real + loss_d_fake
                opt_d.step()

                # update generator
                self.generator.zero_grad()
                x_fake = self.generator(z)
                d_x_fake = self.discriminator(x_fake)
                loss_g = criterion(d_x_fake, t_real)
                loss_g.backward()
                opt_g.step()

                print('%.1f[s]  loss_d: %.3f  loss_g: %.3f' %
                      (time.time()-start, loss_d.item(), loss_g.item()),
                      end='\r')

            if epoch % save_steps == 0:
                z = torch.randn((data_loader.batch_size,
                                 self.generator.input_dim))
                x = self.generate(z)

                dst_path = os.path.join(logdir,
                                        'epoch_%d.png' % epoch)
                self.visualize(dst_path, x)

                torch.save(self.generator.state_dict(),
                           os.path.join(logdir, 'generator_%d.pth' % epoch))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(logdir, 'discriminator_%d.pth' % epoch))

    def generate(self, z):
        with torch.no_grad():
            if not isinstance(z, torch.Tensor):
                _z = torch.Tensor(z)
            else:
                _z = z
            _z = _z.to(self.device)
            return self.generator(_z).detach()

    def visualize(self, dst_path, x):
        n = int(np.sqrt(len(x)))
        x = x[:n**2]
        x = np.transpose(x, (0, 2, 3, 1))
        h, w, c = x.shape[1:]
        x = x.reshape(n, n, *x.shape[1:])
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(n*h, n*w, c)

        if c == 1:
            x = np.squeeze(x, -1)

        x = (x + 1) / 2 * 255
        x = x.numpy().astype('uint8')
        image = Image.fromarray(x)
        image.save(dst_path)
