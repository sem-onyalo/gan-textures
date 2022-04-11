import csv
import io
import json
import logging
import math
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from numpy import mean
from tqdm import tqdm

from storage.manager import StorageManager

MAX_EVAL_SAMPLE_COUNT = 100

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class MetricsMonitor:
    header = ["epoch", "epochs", "batch", "batches", "d_loss", "g_loss", "d_x", "d_g_z1", "d_g_z2"]
    def __init__(self) -> None:
        self.metrics = list()

    def clear(self):
        self.metrics.clear()

    def add_metrics(self, epoch, epochs, batch, batches, d_loss, g_loss, d_x, d_g_z1, d_g_z2):
        self.metrics.append([epoch, epochs, batch, batches, d_loss, g_loss, d_x, d_g_z1, d_g_z2])
        # logging.info(f"{epoch}/{epochs} {batch}/{batches} D_Loss: {d_loss:.4f} D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}")

    def get_losses(self):
        d_losses = [m[4] for m in self.metrics]
        g_losses = [m[5] for m in self.metrics]

        return d_losses, g_losses

class DCGAN_1024:
    def __init__(
        self,
        seed,
        ngpu,
        data_root,
        data_source,
        data_target,
        dataloader_workers,
        epochs,
        batch_size,
        learning_rate,
        adam_beta_1,
        adam_beta_2,
        image_size,
        image_channels,
        g_latent_vector_size,
        g_feature_map_filters,
        g_conv_kernel_size,
        g_conv_stride,
        d_feature_map_filters,
        d_conv_kernel_size,
        d_conv_stride,
        d_activation_negative_slope,
        eval_sample_count,
        eval_epoch_frequency
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.latent_vector_size = g_latent_vector_size
        self.eval_sample_count = eval_sample_count if eval_sample_count <= MAX_EVAL_SAMPLE_COUNT else MAX_EVAL_SAMPLE_COUNT
        self.eval_epoch_frequency = eval_epoch_frequency

        self.timestamp = datetime.utcnow()
        self.metrics_monitor = MetricsMonitor()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.training_data_path = os.path.join(data_root, data_target, self.timestamp.strftime("%Y%m%dT%H%M%SZ"))
        self.storage_manager = StorageManager(self.training_data_path)

        self.write_training_parameters(
            self.timestamp.strftime("%Y%m%dT%H%M%SZ"),
            seed,
            ngpu,
            data_root,
            data_source,
            data_target,
            dataloader_workers,
            epochs,
            batch_size,
            learning_rate,
            adam_beta_1,
            adam_beta_2,
            image_size,
            image_channels,
            g_latent_vector_size,
            g_feature_map_filters,
            g_conv_kernel_size,
            g_conv_stride,
            d_feature_map_filters,
            d_conv_kernel_size,
            d_conv_stride,
            d_activation_negative_slope,
            eval_sample_count,
            eval_epoch_frequency
        )

        logging.info("Building dataloader")
        source_data_path = os.path.join(data_root, data_source)
        self.dataloader = self.build_dataloader(source_data_path, image_size, batch_size, dataloader_workers)
        self.write_samples(self.dataloader.dataset, "target.png", convert_fn=self.dataset_to_plot)

        logging.info("Building generator")
        self.generator = Generator_1024(ngpu, image_channels, g_latent_vector_size, g_feature_map_filters, g_conv_kernel_size, g_conv_stride)
        if self.device.type == "cuda" and ngpu > 1:
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))
        self.generator.apply(weights_init)
        logging.debug(f"Generator:\n{self.generator}")

        logging.info("Building discriminator")
        self.discriminator = Discriminator_1024(ngpu, image_channels, d_feature_map_filters, d_conv_kernel_size, d_conv_stride, d_activation_negative_slope)
        if self.device.type == "cuda" and ngpu > 1:
            self.discriminator = nn.DataParallel(self.discriminator, list(range(ngpu)))
        self.discriminator.apply(weights_init)
        logging.debug(f"Discriminator:\n{self.discriminator}")

        logging.info(f"Training data directory: {self.training_data_path}")
        logging.info("Model init complete")
        logging.info("-" * 50)

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

    def train(self):
        real_label = 1.
        fake_label = 0.

        logging.info("Initializing optimizers")
        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.adam_beta_1, self.adam_beta_2))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.adam_beta_1, self.adam_beta_2))

        logging.info("Generating initial samples")
        eval_latent_vector = torch.randn(64, self.latent_vector_size, 1, 1, device=self.device)
        with torch.no_grad():
            fake = self.generator(eval_latent_vector).detach().cpu()
        self.write_samples(fake, "initial.png", convert_fn=self.prediction_to_plot)

        self.metrics_monitor.clear()
        train_start_date_time_utc = datetime.utcnow()

        for epoch in range(self.epochs):
            d_loss_mean = []
            g_loss_mean = []

            stream = tqdm(self.dataloader)
            for batch, data in enumerate(stream):

                self.discriminator.zero_grad()

                # Train discriminator with real images
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                output = self.discriminator(real_cpu).view(-1)
                d_err_real = criterion(output, label)
                d_err_real.backward()
                d_x_real = output.mean().item()

                # Train discriminator with fake (generated) images
                latent_vector = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)
                fake = self.generator(latent_vector)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                d_err_fake = criterion(output, label)
                d_err_fake.backward()
                d_x_fake = output.mean().item()

                # Update discriminator
                d_err = d_err_real + d_err_fake
                d_optimizer.step()

                # Train and update generator
                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                g_err = criterion(output, label)
                g_err.backward()
                g_x = output.mean().item()
                g_optimizer.step()

                d_loss = d_err.item()
                g_loss = g_err.item()
                d_loss_mean.append(d_loss)
                g_loss_mean.append(g_loss)

                self.metrics_monitor.add_metrics(epoch + 1, self.epochs, batch + 1, len(self.dataloader), d_loss, g_loss, d_x_real, d_x_fake, g_x)

                stream.set_description(f"[{epoch + 1}/{self.epochs}] d_loss: {mean(d_loss_mean):.2f}, g_loss: {mean(g_loss_mean):.2f}")

            logging.info(f"Epoch {epoch + 1} elapsed time: {datetime.utcnow() - train_start_date_time_utc}")

            if (epoch + 1) % self.eval_epoch_frequency == 0 or (epoch + 1) == self.epochs:
                with torch.no_grad():
                    fake = self.generator(eval_latent_vector).detach().cpu()
                self.write_samples(fake, "generated.png", epoch=(epoch + 1), convert_fn=self.prediction_to_plot)
                self.write_metrics(epoch + 1)
                # TODO: write model weights to storage

    def dataset_to_plot(self, samples, sample_count):
        plots = list()
        for i in range(sample_count):
            sample = samples[i][0]
            sample = (sample + 1) / 2.0 # scale from -1,1 to 0,1
            sample = sample.permute(1, 2, 0)
            plots.append(sample)
        return plots

    def prediction_to_plot(self, samples, sample_count):
        plots = list()
        for i in range(sample_count):
            sample = samples[i]
            sample = (sample + 1) / 2.0 # scale from -1,1 to 0,1
            sample = sample.permute(1, 2, 0)
            plots.append(sample)
        return plots

    def write_samples(self, samples, filename, fig_dim=20, epoch=None, convert_fn=None):
        sample_count = self.eval_sample_count if self.eval_sample_count <= len(samples) else len(samples)
        grid_dim = math.floor(math.sqrt(sample_count))

        if convert_fn != None:
            samples = convert_fn(samples, grid_dim ** 2)

        plt.figure(figsize=(fig_dim, fig_dim))
        for i in range(grid_dim ** 2):
            plt.subplot(grid_dim, grid_dim, i + 1)
            plt.axis("off")
            plt.imshow(samples[i]) # TODO: set gray based on function param

        buffer = io.BytesIO()
        plt.savefig(buffer)
        plt.close()

        path_parts = list()
        if epoch != None:
            path_parts.append("epoch")
            path_parts.append(str(epoch))

        self.storage_manager.write_bytes(filename, buffer.getvalue(), path_parts=path_parts)

    def write_metrics(self, epoch):
        path_parts = ["epoch", str(epoch)]

        d_losses, g_losses = self.metrics_monitor.get_losses()

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss")
        plt.plot(g_losses, label="G-Loss")
        plt.plot(d_losses, label="D-Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()

        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer)
        plt.close()
        plot_buffer.seek(0)

        self.storage_manager.write_bytes("loss.png", plot_buffer.getvalue(), path_parts=path_parts)

        metrics_buffer = io.StringIO()
        writer = csv.writer(metrics_buffer)
        writer.writerow(self.metrics_monitor.header)
        writer.writerows(self.metrics_monitor.metrics)

        self.storage_manager.write_bytes("metrics.csv", metrics_buffer.getvalue().encode(), path_parts=path_parts)

    def write_training_parameters(
        self,
        timestamp,
        seed,
        ngpu,
        data_root,
        data_source,
        data_target,
        dataloader_workers,
        epochs,
        batch_size,
        learning_rate,
        adam_beta_1,
        adam_beta_2,
        image_size,
        image_channels,
        g_latent_vector_size,
        g_feature_map_filters,
        g_conv_kernel_size,
        g_conv_stride,
        d_feature_map_filters,
        d_conv_kernel_size,
        d_conv_stride,
        d_activation_negative_slope,
        eval_sample_count,
        eval_epoch_frequency
    ):
        training_parameters = {
            "timestamp": timestamp,
            "seed": seed,
            "ngpu": ngpu,
            "data_root": data_root,
            "data_source": data_source,
            "data_target": data_target,
            "dataloader_workers": dataloader_workers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "adam_beta_1": adam_beta_1,
            "adam_beta_2": adam_beta_2,
            "image_size": image_size,
            "image_channels": image_channels,
            "g_latent_vector_size": g_latent_vector_size,
            "g_feature_map_filters": g_feature_map_filters,
            "g_conv_kernel_size": g_conv_kernel_size,
            "g_conv_stride": g_conv_stride,
            "d_feature_map_filters": d_feature_map_filters,
            "d_conv_kernel_size": d_conv_kernel_size,
            "d_conv_stride": d_conv_stride,
            "d_activation_negative_slope": d_activation_negative_slope,
            "eval_sample_count": eval_sample_count,
            "eval_epoch_frequency": eval_epoch_frequency
        }
        
        buffer = json.dumps(training_parameters, indent=4).encode()
        self.storage_manager.write_bytes("training_parameters.json", buffer)

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
