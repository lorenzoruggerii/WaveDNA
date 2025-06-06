"""Launch ResNet pipeline starting from experiment name and TF"""

import os
import sys
import argparse
import wget
import subprocess
import glob

BED_FOLDER = "/where/to/put/bedfile"
FASTA_FOLDER = "/where/to/put/fastafile"
GENOME = "/path/to/ref_genome"
NEG_FOLDER = "/path/to/DHS"
IMAGE_FOLDER = "/where/to/put/images"
SPLITS = ['train', 'test']
CLASSES = ['positive', 'negative']

def download_data(code: str, tf: str):
    print(f"\nDownloading {code}...\n")
    get_download_link = lambda code: f"https://www.encodeproject.org/files/{code}/@@download/{code}.bed.gz"
    outdir = os.path.join(BED_FOLDER, tf)
    os.makedirs(outdir, exist_ok=True)
    filename = wget.download(url=get_download_link(code), out=outdir)
    # gzip file
    cmd = f"gzip -d {filename}"
    ec = subprocess.call(cmd, shell=True)
    assert ec == 0, f"Error with {cmd}"
    print(f"\n{code} bed downloaded in {outdir}!\n")

def split_dataset(tf: str):
    print(f"\nSplitting datasets for {tf}...\n")
    outdir = os.path.join(FASTA_FOLDER, tf)
    os.makedirs(outdir, exist_ok=True)
    pos_folder = os.path.join(BED_FOLDER, tf)
    cmd = f"python src/dataset_split.py {pos_folder} {NEG_FOLDER} {GENOME} {outdir}"
    ec = subprocess.call(cmd, shell=True)
    assert ec == 0, f"Problem with {cmd}"
    print(f"\n{tf} dataset splt successfully!\n")

def create_images(tf: str):
    print(f"\nCreating images for {tf}...\n")
    image_dir = os.path.join(IMAGE_FOLDER, tf)
    os.makedirs(image_dir, exist_ok=True)
    for split in SPLITS:
        for c in CLASSES:
            fasta_dir = os.path.join(FASTA_FOLDER, tf, f"{split}data", c)
            fasta_lists = glob.glob("*.fa", root_dir=fasta_dir)
            fasta_path = os.path.join(fasta_dir, fasta_lists[0])
            outdir = os.path.join(image_dir, split, c)
            prefix = f"{split}_{c[:3]}"
            cmd = f"python src/create_colormaps.py {fasta_path} {outdir} 0 {prefix} 100000 16"
            ec = subprocess.call(cmd, shell=True)
            assert ec == 0, f"Error in {cmd}"
            print(f"\n{split} {c} images for {tf} created successfully\n")

def run_model(tf: str):
    print(f"\nRunning model for {tf}...\n")
    cmd = f"python src/train_resnet.py --tf_name {tf} --batch_size 4"
    ec = subprocess.call(cmd, shell=True)
    assert ec == 0, f"Error in {cmd}"
    print(f"Model {tf} trained successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, required=True, help="TF selected to run the model on.")
    parser.add_argument("--bedCode", type=str, required=True, help="ID of the bedfile.")
    parser.add_argument("--run_model", type=int, required=True, help="Whether to run the model or not.")

    args = parser.parse_args()

    # 1. Download data from encode and put in the right folder
    download_data(code=args.bedCode, tf=args.tf_name)
    # 2. Run dataset split and create fasta files
    split_dataset(tf=args.tf_name)
    # 3. Run create_colormaps to create images and put in the right folder
    create_images(tf=args.tf_name)
    # 4. Run the training of the model
    if args.run_model:
        run_model(tf=args.tf_name)

if __name__ == "__main__":
    main()
