import logging
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN_1024:
    def __init__(
        self,
        seed,
        ngpu,
        data_root,
        dataloader_workers,
        epochs,
        batch_size,
        learning_rate,
        adam_beta_1,
        image_size,
        image_channels,
        g_latent_vector_size,
        g_feature_map_filters,
        g_conv_kernel_size,
        g_conv_stride,
        d_feature_map_filters,
        d_conv_kernel_size,
        d_conv_stride,
        d_activation_negative_slope
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.dataloader = self.build_dataloader(data_root, image_size, batch_size, dataloader_workers)
        self.plot_samples(self.tensors_to_plots(self.dataloader.dataset), "data/temp.png")

        self.generator = Generator_1024(ngpu, image_channels, g_latent_vector_size, g_feature_map_filters, g_conv_kernel_size, g_conv_stride)
        if self.device.type == "cuda" and ngpu > 1:
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))
        self.generator.apply(weights_init)
        logging.info(f"Generator:\n{self.generator}")

        self.discriminator = Discriminator_1024(ngpu, image_channels, d_feature_map_filters, d_conv_kernel_size, d_conv_stride, d_activation_negative_slope)
        if self.device.type == "cuda" and ngpu > 1:
            self.discriminator = nn.DataParallel(self.discriminator, list(range(ngpu)))
        self.discriminator.apply(weights_init)
        logging.info(f"Discriminator:\n{self.discriminator}")

    def build_dataloader(self, data_root, image_size, batch_size, workers):
        dataset = datasets.ImageFolder(
            root=data_root,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )

        return dataloader

    def tensors_to_plots(self, samples, grid_dim=10):
        plots = list()
        for i in range(grid_dim * grid_dim):
            sample = samples[i][0]
            sample = (sample + 1) / 2.0 # scale from -1,1 to 0,1
            sample = sample.permute(1, 2, 0)
            plots.append(sample)
        return plots
    
    def plot_samples(self, samples, target_path, grid_dim=10, fig_dim=20):
        plt.figure(figsize=(fig_dim, fig_dim))
        for i in range(grid_dim * grid_dim):
            plt.subplot(grid_dim, grid_dim, i + 1)
            plt.axis("off")
            plt.imshow(samples[i], cmap="gray_r") # TODO: set gray based on function param
        
        plt.savefig(target_path)
        plt.close()

class Generator_1024(nn.Module):
    def __init__(self, ngpu, output_channels, latent_vector_size, feature_map_filters, kernel_size, stride):
        super(Generator_1024, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 100x1 -> 4x4x(feature_map_filters * 128)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(latent_vector_size, feature_map_filters * 128, kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_filters * 128),
            nn.ReLU(True),
            # 4x4x(feature_map_filters * 128) -> 8x8x(feature_map_filters * 64)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 128, feature_map_filters * 64, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 64),
            nn.ReLU(True),
            # 8x8x(feature_map_filters * 64) -> 16x16x(feature_map_filters * 32)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 64, feature_map_filters * 32, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 32),
            nn.ReLU(True),
            # 16x16x(feature_map_filters * 32) -> 32x32x(feature_map_filters * 16)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 32, feature_map_filters * 16, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 16),
            nn.ReLU(True),
            # 32x32x(feature_map_filters * 16) -> 64x64x(feature_map_filters * 8)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 16, feature_map_filters * 8, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 8),
            nn.ReLU(True),
            # 64x64x(feature_map_filters * 8) -> 128x128x(feature_map_filters * 4)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 8, feature_map_filters * 4, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 4),
            nn.ReLU(True),
            # 128x128x(feature_map_filters * 4) -> 256x256x(feature_map_filters * 2)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 4, feature_map_filters * 2, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 2),
            nn.ReLU(True),
            # 256x256x(feature_map_filters * 2) -> 512x512x(feature_map_filters)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters * 2, feature_map_filters, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters),
            nn.ReLU(True),
            # 512x512x(feature_map_filters) -> 1024x1024x(output_channels)  (when defaults set: kernel_size==4, stride==2)
            nn.ConvTranspose2d(feature_map_filters, output_channels, kernel_size, stride, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_1024(nn.Module):
    def __init__(self, ngpu, input_channels, feature_map_filters, kernel_size, stride, activation_negative_slope):
        super(Discriminator_1024, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 1024x1024xinput_channels -> 512x512x(feature_map_filters) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(input_channels, feature_map_filters, kernel_size, stride, padding=1, bias=False),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 512x512x(feature_map_filters) -> 256x256x(feature_map_filters * 2) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters, feature_map_filters * 2, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 2),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 256x256x(feature_map_filters * 2) -> 128x128x(feature_map_filters * 4) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 2, feature_map_filters * 4, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 4),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 128x128x(feature_map_filters * 4) -> 64x64x(feature_map_filters * 8) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 4, feature_map_filters * 8, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 8),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 64x64x(feature_map_filters * 8) -> 32x32x(feature_map_filters * 16) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 8, feature_map_filters * 16, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 16),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 32x32x(feature_map_filters * 16) -> 16x16x(feature_map_filters * 32) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 16, feature_map_filters * 32, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 32),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 16x16x(feature_map_filters * 32) -> 8x8x(feature_map_filters * 64) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 32, feature_map_filters * 64, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 64),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 8x8x(feature_map_filters * 64) -> 4x4x(feature_map_filters * 128) (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 64, feature_map_filters * 128, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_filters * 128),
            nn.LeakyReLU(activation_negative_slope, inplace=True),
            # 4x4x(feature_map_filters * 128) -> 4x4x1 (when defaults set: kernel_size==4, stride==2)
            nn.Conv2d(feature_map_filters * 128, 1, kernel_size, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
