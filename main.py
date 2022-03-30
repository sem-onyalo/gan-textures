import argparse
import logging

from model import DCGAN_1024

def init_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="data/01-cur")
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--adam_beta_1", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=1080)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--g_latent_vector_size", type=int, default=100)
    parser.add_argument("--g_feature_map_filters", type=int, default=64)
    parser.add_argument("--g_conv_kernel_size", type=int, default=4)
    parser.add_argument("--g_conv_stride", type=int, default=2)
    parser.add_argument("--d_feature_map_filters", type=int, default=64)
    parser.add_argument("--d_conv_kernel_size", type=int, default=4)
    parser.add_argument("--d_conv_stride", type=int, default=2)
    parser.add_argument("--d_activation_negative_slope", type=float, default=0.2)
    args = parser.parse_args()

    init_logger()

    gan = DCGAN_1024(
        args.seed,
        args.ngpu,
        args.data_root,
        args.dataloader_workers,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.adam_beta_1,
        args.image_size,
        args.image_channels,
        args.g_latent_vector_size,
        args.g_feature_map_filters,
        args.g_conv_kernel_size,
        args.g_conv_stride,
        args.d_feature_map_filters,
        args.d_conv_kernel_size,
        args.d_conv_stride,
        args.d_activation_negative_slope
    )
