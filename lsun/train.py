import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms
import os
import sys
sys.path.append(os.getcwd())
from mnist.models import Generator, Discriminator
from dcgan import DCGAN


def main():
    parser = argparse.ArgumentParser(description='LSUN')
    parser.add_argument('image_dir')
    parser.add_argument('--batch_size', '-bs',
                        type=int, default=64)
    parser.add_argument('--nb_epoch', '-e',
                        type=int, default=100)
    parser.add_argument('--save_steps', '-ss',
                        type=int, default=10)
    parser.add_argument('--lr_d',
                        type=float, default=1e-3)
    parser.add_argument('--lr_g',
                        type=float, default=1e-3)
    parser.add_argument('--ngf', '-ngf',
                        type=int, default=64)
    parser.add_argument('--ndf', '-ndf',
                        type=int, default=64)
    parser.add_argument('--latent_dim', '-ld',
                        type=int, default=128)
    parser.add_argument('--logdir',
                        type=str, default='logs')
    parser.add_argument('--no-cuda',
                        dest='use_cuda',
                        action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.LSUN(args.image_dir,
                      classes=['bedroom_train'],
                      transform=transforms.Compose([
                          transforms.Resize(64),
                          transforms.ToTensor(),
                          transforms.Lambda(lambda x: (x-0.5)*2)
                       ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    generator = Generator(args.latent_dim, args.ngf)
    discriminator = Discriminator(args.ndf)
    gan = DCGAN(generator, discriminator, device)
    gan.init_params()

    gan.fit(train_loader,
            nb_epoch=args.nb_epoch,
            lr_d=args.lr_d,
            lr_g=args.lr_g,
            save_steps=args.save_steps,
            logdir=args.logdir)


if __name__ == '__main__':
    main()
