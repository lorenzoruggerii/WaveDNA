# WaveDNA

**A Deep Learning Tool for DNA Sequence Classification Using Wavelet Transform and Convolutional Neural Networks**

## Overview

WaveDNA is a novel computational tool that transforms DNA sequences into image-like representations for deep learning-based classification. By leveraging the power of wavelet transforms and convolutional neural networks, WaveDNA enables accurate identification of transcription factor binding sites (TFBSs) and other sequence patterns in genomic data.

## Key Features

- **Wavelet-based Encoding**: Converts DNA sequences into two-dimensional frequency representations using Continuous Wavelet Transform (CWT)
- **Deep Learning Classification**: Utilizes ResNet50 architecture for robust sequence classification
- **Interpretability studies**: Captures biologically relevant sequence parts leveraging Grad-CAM

## How It Works

WaveDNA operates through a two-stage pipeline:

### 1. Sequence Encoding
- **Numerical Mapping**: DNA nucleotides are converted to numerical values (A=1, C=2, G=3, T=4)
- **Wavelet Transform**: The numerical signal undergoes Continuous Wavelet Transform using the Morlet wavelet
- **Image Generation**: Wavelet coefficients are visualized as 2D images with color-coded intensity patterns

### 2. Deep Learning Classification
- **CNN Processing**: Generated wavelet images are fed into a ResNet50 neural network
- **Feature Extraction**: The 50-layer architecture with residual blocks extracts hierarchical features
- **Binary Output**: Final classification into TFBS (positive) or non-TFBS (negative) categories

## Data for reproducing the analysis
This project uses ChIP-seq data from the [ENCODE Project](https://www.encodeproject.org/) to train and evaluate models, as well as to perform explainability analyses.

The following table lists the 13 transcription factors used in this project, along with their corresponding BED and BigWig file identifiers from ENCODE. The BED files are used to extract positive sequences for training WaveDNA, while bigWig files are used for conducting explainability analysis. For negative sequences, we used Meuleman et al. [DHS index](https://www.encodeproject.org/annotations/ENCSR857UZV/).

| Transcription Factor (TF) | BED ID       | BigWig ID    |
|---------------------------|--------------|--------------|
| SETDB1                   | ENCFF812YDP  | ENCFF413AIX  |
| GATA1                    | ENCFF657CTC  | ENCFF331URE  |
| GATA2                    | ENCFF409YKQ  | ENCFF184LTK  |
| JUND                     | ENCFF658HLG  | ENCFF292YDK  |
| RFX5                     | ENCFF438ZEN  | ENCFF482YCH  |
| HOXA3                    | ENCFF753OCA  | ENCFF631HAF  |
| TCF7L2                   | ENCFF726SWQ  | ENCFF406WNX  |
| ZBED1                    | ENCFF753UAX  | ENCFF606OSZ  |
| NFKB2                    | ENCFF687TJX  | ENCFF918FFX  |
| MEF2B                    | ENCFF884QQW  | ENCFF710GGG  |
| LCORL                    | ENCFF606UCO  | ENCFF227FSQ  |
| MBD2                     | ENCFF788YHU  | ENCFF510BMM  |
| GMEB1                    | ENCFF365ETH  | ENCFF398IIR  |

### Example

To download the `ENCFF812YDP` BED file, use:
https://www.encodeproject.org/files/ENCFF812YDP/@@download/ENCFF812YDP.bed.gz

Please make sure to **unzip** your BED file before running the analysis.

To download the `ENCFF413AIX` bigWig file, use:
https://www.encodeproject.org/files/ENCFF812YDP/@@download/ENCFF812YDP.bigWig

## Usage

### 1. Environment Setup

First, create a conda/mamba environment with Python 3.10 and install the required dependencies:

```bash
# Create a new environment with Python 3.10
conda create -n wavedna python=3.10
# or using mamba (faster)
mamba create -n wavedna python=3.10

# Activate the environment
conda activate wavedna

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Data Preparation

#### Download Reference Genome

Download the human reference genome (hg38) from UCSC:

```bash
# Create data directory
mkdir -p data/genome

# Download hg38 reference genome
cd data/genome
wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz

# Extract the genome file
gunzip hg38.fa.gz
```

#### Prepare Input Data

WaveDNA requires two types of input data:

1. **Positive sequences**: BED file from ChIP-seq experiments (ENCODE database) (see Section `Data` to download data used in the study)
   - Contains genomic coordinates of transcription factor binding sites
   - Format: `chr`, `start`, `end`, `name`, `score`, `strand`

2. **Negative sequences**: BED file containing DNase hypersensitive sites
   - We recommend using data from Meuleman et al. (available at [https://www.encodeproject.org/annotations/ENCSR857UZV/])
   - These represent accessible chromatin regions without specific TF binding

```bash
# Create directories for input data
mkdir -p data/bed_files

# Example file structure:
# data/bed_files/positive_sequences.bed  # ChIP-seq peaks from ENCODE
# data/bed_files/negative_sequences.bed  # DNase hypersensitive sites
```

#### BED File Format Requirements:
Your BED files should contain at least 10 columns with peak summit information:
```
chr1    1000    1200    peak1    100    +    1.5    10.2    50    100
```
- Columns 1-3: chromosome, start, end
- Column 7: enrichment score (for duplicate removal)
- Column 10: peak summit position (for centering)


### 3. Dataset Splitting

Run the dataset split script to create training and test sets from your ChIP-seq data:

```bash
python dataset_split.py positive_bed_folder negative_bed_folder data/genome/hg38.fa data/processed
```

#### Input Structure:
Organize your input data as follows:
```
positive_bed_folder/
├── TF1_peaks.bed
├── TF2_peaks.bed
└── TF3_peaks.bed

negative_bed_folder/
└── DNase_sites.bed  # Single negative dataset file
```

#### Output Structure:
The script generates the following directory structure:
```
data/processed/
├── traindata/
│   ├── positive/
│   │   ├── TF1_train.fa
│   │   ├── TF2_train.fa
│   │   └── TF3_train.fa
│   └── negative/
│       ├── TF1_neg_train.fa
│       ├── TF2_neg_train.fa
│       └── TF3_neg_train.fa
└── testdata/
    ├── positive/
    │   ├── TF1_test.fa
    │   ├── TF2_test.fa
    │   └── TF3_test.fa
    └── negative/
        ├── TF1_neg_test.fa
        ├── TF2_neg_test.fa
        └── TF3_neg_test.fa
```

#### Command Arguments:
1. `positive_bed_folder`: Directory containing ChIP-seq BED files (one per transcription factor)
2. `negative_bed_folder`: Directory containing a single BED file with DNase hypersensitive sites
3. `genome_file`: Path to reference genome FASTA file (e.g., `hg38.fa`)
4. `output_directory`: Directory where processed datasets will be saved


### 4. Wavelet Transform and Colormap Generation

Convert DNA sequences to wavelet-based image representations using the continuous wavelet transform:

```bash
# Process positive training sequences
python create_colormaps.py \
    data/processed/traindata/positive/TF1_train.fa \
    data/processed/images/train/positive \
    0 \
    TF1_pos \
    1000 \
    8

# Process negative training sequences  
python create_colormaps.py \
    data/processed/traindata/negative/TF1_neg_train.fa \
    data/processed/images/train/negative \
    0 \
    TF1_neg \
    1000 \
    8

# Process positive test sequences
python create_colormaps.py \
    data/processed/testdata/positive/TF1_test.fa \
    data/processed/images/test/positive \
    0 \
    TF1_pos \
    500 \
    8

# Process negative test sequences
python create_colormaps.py \
    data/processed/testdata/negative/TF1_neg_test.fa \
    data/processed/images/test/negative \
    0 \
    TF1_neg \
    500 \
    8
```

#### Command Arguments:
1. `fasta_file`: Path to input FASTA file containing DNA sequences
2. `output_directory`: Directory where colormap images will be saved
3. `synchrosqueeze`: Use synchrosqueezing transform (1) or standard CWT (0)
4. `prefix`: Prefix for output image filenames
5. `threshold`: Maximum number of sequences to process (for limiting dataset size)
6. `num_workers`: Number of CPU cores for parallel processing

#### Transform Options:
- **Standard CWT** (`synchrosqueeze=0`): Uses continuous wavelet transform coefficients
- **Synchrosqueezed CWT** (`synchrosqueeze=1`): Enhanced frequency localization using synchrosqueezing


### 5. ResNet50 Training and Classification

Train the ResNet50 convolutional neural network on the generated wavelet images:

#### Prerequisites:
Before training, ensure your image directory structure matches the expected format:
```
data/processed/images/
├── TF_NAME/
│   ├── train/
│   │   ├── positive/
│   │   │   ├── TF1_pos_cmap_0.png
│   │   │   └── ...
│   │   └── negative/
│   │       ├── TF1_neg_cmap_0.png
│   │       └── ...
│   └── test/
│       ├── positive/
│       │   ├── TF1_pos_cmap_0.png
│       │   └── ...
│       └── negative/
│           ├── TF1_neg_cmap_0.png
│           └── ...
```

#### Training Command:
```bash
# Train ResNet50 for a specific transcription factor
python train_resnet.py --tf_name PATZ1 --batch_size 16

# Example for different transcription factors
python train_resnet.py --tf_name CTCF --batch_size 32
python train_resnet.py --tf_name MAX --batch_size 8
```

#### Command Arguments:
- `--tf_name`: Name of the transcription factor (must match directory name)
- `--batch_size`: Training batch size (adjust based on GPU memory)

#### Training Configuration:
The model uses the following default settings:
- **Architecture**: ResNet50 with pre-trained ImageNet weights
- **Fine-tuning**: Full model fine-tuning with adaptive learning rate
- **Input size**: Images resized to 224×224 pixels
- **Normalization**: ImageNet standard normalization
- **Optimizer**: AdamW with learning rate 3e-5
- **Scheduler**: ReduceLROnPlateau based on validation accuracy
- **Loss function**: CrossEntropyLoss for binary classification
- **Epochs**: 3 (default, can be modified in code)

#### Output:
After training, the following files are generated:
```
models/ResNet/TF_NAME/
└── ResNet_TF_NAME_1bs_fullft.pth  # Best model checkpoint
```

The script uses wandb to log the metrics during model's training. Please make sure to log in into your wandb account to track the run.

### 6. Model Interpretation with Grad-CAM

After training, analyze and interpret model predictions using Grad-CAM (Gradient-weighted Class Activation Mapping) to understand which regions of the wavelet images contribute most to the classification decisions and correlate them with biological ChIP-seq signals:

#### XAI Grad-CAM Analysis

The `xai_grad_cam.py` script provides comprehensive model interpretability by generating activation heatmaps and correlating them with experimental ChIP-seq data to validate biological relevance of learned features.

#### Prerequisites:
Before running Grad-CAM analysis, ensure you have:
- **Trained model**: A saved ResNet50 model checkpoint from the training step
- **Test images**: Wavelet-transformed images from your test dataset
- **ChIP-seq data**: BigWig file containing experimental binding signals
- **Genomic coordinates**: BED file with coordinates matching your test sequences

#### Command Usage:
```bash
# Basic Grad-CAM analysis for a specific transcription factor
python xai_grad_cam.py \
    -indir data/processed/images/test/positive \
    -outdir results/gradcam/CTCF \
    -model models/ResNet/CTCF/ResNet_CTCF_1bs_fullft.pth \
    -bigwig data/chipseq/CTCF_signal.bw \
    -bedfile data/bed_files/CTCF_test_sequences.bed \
    -layer 4 \
    -convIndex 1 \
    -blockIndex 2
```

#### Command Arguments:
- `-indir`: Directory containing test images (wavelet-transformed sequences)
- `-outdir`: Output directory for Grad-CAM visualizations and results
- `-model`: Path to trained ResNet50 model checkpoint
- `-bigwig`: Path to BigWig file containing ChIP-seq signal values
- `-bedfile`: Path to BED file with genomic coordinates of test sequences
- `-layer`: ResNet layer to analyze (1-4, corresponding to layer1-layer4)
- `-convIndex`: Convolutional layer index within the block (1-3)
- `-blockIndex`: Block index within the specified layer (0-based)
- `-max_sequences`: Optional limit on number of sequences to process

#### Layer Selection Guide:
ResNet50 architecture provides different levels of feature abstraction (from Layer1 to Layer4)
Each layer contains multiple blocks with convolutional filters that can be analyzed independently.

## 7. End-to-End Pipeline Script

To simplify the entire workflow — from data acquisition to model training — WaveDNA provides a streamlined script for launching the full ResNet-based classification pipeline with minimal manual intervention.

### Script: `ResNet_pipeline.py`

This script automates the following steps:
1. **Download ChIP-seq BED file** from ENCODE based on the provided accession code.
2. **Split the dataset** into training and testing FASTA files.
3. **Generate wavelet-based images** from the FASTA sequences.
4. **Train the ResNet50 model** on the generated images.

### Usage

```bash
python run_pipeline.py --tf_name CTCF --bedCode ENCFF001ABC --run_model 1
```

Arguments:

--`tf_name`: Name of the transcription factor (used for folder naming and labeling)

--`bedCode`: ENCODE accession ID for the ChIP-seq BED file

--`run_model`: Set to 1 to train the model after preprocessing, 0 to skip training
