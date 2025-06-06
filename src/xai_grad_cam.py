import os
import argparse
import torch as t
import numpy as np
import pyBigWig
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from imageio.v2 import get_writer, imread
from typing import List
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils import get_resnet_for_fine_tuning
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import random

# Constants
BASES = ['A', 'C', 'T', 'G']
GKMPREDICT = "gkmpredict"
RC = {"A": "T",
      "T": "A",
      "C": "G",
      "G": "C",
      "a": "t",
      "t": "a",
      "c": "g",
      "g": "c"}
IMAGE_SIZE = 224 # image size for ResNet50
device = 'cuda' if t.cuda.is_available() else 'cpu'

# Image transforms
transform_resnet = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device))
])

transform_heatmap = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda x: np.array(x) / 255)
])

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

def create_column_activation_plot_refined(
    grayscale_cam: np.array,
    output_path: str,
    signal_values: np.array,
):
    """Create the activation plot returning the correlation with biological ChIP-seq signal"""
    
    def normalize_array(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    # Compute column-wise activations and normalize
    column_sums = np.sum(grayscale_cam, axis=0)
    column_sums = normalize_array(column_sums)

    # Interpolate to match original sequence length
    x_orig = np.linspace(0, 1, len(column_sums))
    x_new = np.linspace(0, 1, len(signal_values))
    column_sums = np.interp(x_new, x_orig, column_sums)

    if len(signal_values) > len(column_sums):
        signal_values = signal_values[:len(column_sums)]
    else: 
        signal_values = np.pad(signal_values, (0, len(column_sums) - len(signal_values)), mode = 'constant')

    # Repeat to make it bigger
    column_sums_img = np.repeat(column_sums.reshape(1, -1), 7, axis=0)

    # Create main figure and heatmap axis
    fig, ax_heatmap = plt.subplots(figsize=(14, 4))

    # Create divider
    divider = make_axes_locatable(ax_heatmap)

    # Top axis for signal
    ax_signal = divider.append_axes("top", size="100%", pad=0.05, sharex=ax_heatmap)

    # --- Plot heatmap ---
    im = ax_heatmap.imshow(column_sums_img, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax_heatmap.yaxis.set_visible(False)

    # Colorbar
    cax = divider.append_axes('right', size='2%', pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1])

    # --- Plot signal ---
    ax_signal.plot(signal_values, color='blue', linewidth=1)
    ax_signal.fill_between(
        np.arange(len(signal_values)),  # x values
        signal_values,                  # y values
        color='blue',                   # fill color
        alpha=0.3                       # transparency
    )
    ax_signal.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate correlation
    corr_r, pvalue_r = pearsonr(signal_values, column_sums)

    return (corr_r, pvalue_r)


def load_model(model_path: str, num_classes: int = 2):
    """Load and prepare the model for inference."""
    model = get_resnet_for_fine_tuning(num_classes=num_classes)
    model.load_state_dict(t.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_gradcam_image(model, input_tensor: t.Tensor, targets: List, rgb_img_np: np.array, layer):
    """Generate a Grad-CAM heatmap for a given model layer."""
    with GradCAM(model=model, target_layers=[layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        # grayscale_cam = np.where(grayscale_cam > np.percentile(grayscale_cam, 50), grayscale_cam, 0)
        return rgb_img_np * np.expand_dims(grayscale_cam, axis=-1), grayscale_cam

def save_image(image: np.array, output_path: str):
    """Save an image to disk."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gif(image_paths: List[str], output_path: str):
    """Create a GIF from a sequence of images."""
    with get_writer(output_path, mode='I', duration=0.5) as writer:
        for img_path in image_paths:
            writer.append_data(imread(img_path))


def main():
    seed_everything(42)

    p = argparse.ArgumentParser()

    # Parse input args
    p.add_argument('-indir', type=str, required=True, help="input folder containing images")
    p.add_argument('-outdir', type=str, required=True, help="folder in which save Grad-CAM feature maps")
    p.add_argument('-max_sequences', type=int, required=False, help="maximum number of sequences to consider")
    p.add_argument('-model', type=str, required=True, help="path to the trained model")
    p.add_argument('-bigwig', type=str, required=True, help="path to the bigWig file containing ChIP-seq signals")
    p.add_argument('-bedfile', type=str, required=True, help="path to bedfile containing sequences")
    p.add_argument('-layer', type=int, required=True, help="model's layer to investigate via Grad-CAM")
    p.add_argument('-convIndex', type=int, required=True, help="index of the convolutional filter to investigate")
    p.add_argument('-blockIndex', type=int, required=True, help="index of layer's block to investigate")
    args = p.parse_args()

    # Read images
    imagefiles = natsorted(os.listdir(args.indir))

    # Read bedfile for genomic coordinates of the sequences
    bedfile = pd.read_csv(args.bedfile, sep = '\t', header = None)
    testseqs = bedfile[bedfile[0] == "chr2"].sort_values(by = [1])

    # Extract signal values
    bw = pyBigWig.open(args.bigwig)

    # Initialize results df
    results_list = []
    columns = ["chr", "start", "stop", "corr_resnet", "p_value_resnet"]

    if args.max_sequences is not None and (args.max_sequences < len(imagefiles)):
        imagefiles = imagefiles[:args.max_sequences]
    
    # Load model with weights
    model = load_model(args.model)

    # Extract conv layer to investigate
    target_block = getattr(model, f'layer{args.layer}')[args.blockIndex]
    target_conv = getattr(target_block, f'conv{args.convIndex}')
    target_layers = [target_conv]
    targets = [ClassifierOutputTarget(1)] # positive class

    for i, imagefile in enumerate(tqdm(imagefiles, total=len(imagefiles), colour='GREEN')):
        image_path = os.path.join(args.indir, imagefile)
        rgb_img = Image.open(image_path).convert('RGB')
        input_tensor = transform_resnet(rgb_img).unsqueeze(0)
        rgb_img_np = transform_heatmap(rgb_img)

        # Initialize image mean with zeros
        output_image_mean = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        # Read ChIP-seq signal values from bigWig file according to the specific coordinates
        signal_values = bw.values("chr2", testseqs.iloc[i, 1], testseqs.iloc[i, 2])
        
        # Obtain activation map for each target layer
        for idx, layer in enumerate(target_layers):
            output_image, grayscale_cam = generate_gradcam_image(model, input_tensor, targets, rgb_img_np, layer)
            output_image_mean += grayscale_cam
            layer_dir = os.path.join(args.outdir, f"Layer_{idx}")
            os.makedirs(layer_dir, exist_ok=True)
        
        # Compute mean of activation maps and save
        output_image_mean /= len(target_layers)
        output_image_path = os.path.join(layer_dir, f"Pos_activations_{i}.png")
        corr_r, pvalue_r = create_column_activation_plot_refined(output_image_mean, output_image_path, signal_values)

        # Save correlation results
        results_list.append(["chr2", testseqs.iloc[i, 1], testseqs.iloc[i, 2], corr_r, pvalue_r])
        results_df = pd.DataFrame(results_list, columns=columns)
        results_df.to_csv(f"{args.outdir}/results.tsv", sep='\t', index=False)


if __name__ == '__main__':
    main()
