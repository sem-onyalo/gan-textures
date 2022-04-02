import argparse
import cv2 as cv
import logging
import os
import shutil
import uuid
from datetime import datetime

from tqdm import tqdm

CV_Y_DIM = 0
CV_X_DIM = 1

def init_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

def curate(**kwargs):
    src_dir = kwargs["src_dir"]
    tgt_dir = kwargs["tgt_dir"]
    tgt_dim = kwargs["tgt_dim"]
    image_files = kwargs["image_files"]

    if os.path.isdir(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)

    stream = tqdm(image_files)
    for _, image_file in enumerate(stream):
        image_file_path = os.path.join(src_dir, image_file)
        image = cv.imread(image_file_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        crop_start_x = 0
        crop_start_y = 0

        batch_num = 1
        while True:
            image_crop = image[crop_start_y:crop_start_y + tgt_dim, crop_start_x:crop_start_x + tgt_dim]
            tgt_file = os.path.join(tgt_dir, f"{batch_num:04d}-{image_file}")
            stream.set_description(f"source: {image_file}")
            cv.imwrite(tgt_file, image_crop)

            crop_start_x += tgt_dim
            if crop_start_x + tgt_dim > image.shape[CV_X_DIM]:
                crop_start_y += tgt_dim
                if crop_start_y + tgt_dim > image.shape[CV_Y_DIM]:
                    break
                else:
                    crop_start_x = 0
            batch_num += 1

def main(args):
    data_dir = args.data_dir
    data_dir_raw = args.data_dir_raw
    data_dir_cur = args.data_dir_cur
    class_name = args.class_name
    target_dim = args.target_dim

    start_utc = datetime.utcnow()
    images_dir_raw = os.path.join(data_dir, data_dir_raw)
    images_dir_cur = os.path.join(data_dir, data_dir_cur, class_name)
    image_files = os.listdir(images_dir_raw)
    curate(src_dir=images_dir_raw, tgt_dir=images_dir_cur, tgt_dim=target_dim, image_files=image_files)
    logging.info(f"[DONE] elapsed time: {datetime.utcnow() - start_utc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--data_dir_raw", type=str, default="00-raw")
    parser.add_argument("--data_dir_cur", type=str, default="01-cur")
    parser.add_argument("--class_name", type=str, default="concrete")
    parser.add_argument("--target_dim", type=int, default=1080)
    args = parser.parse_args()
    init_logger()
    main(args)
