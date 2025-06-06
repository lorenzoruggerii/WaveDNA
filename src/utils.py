import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

MAPPING_DICT = {"A": 1, "C": 2, "G": 3, "T": 4}
INVERSE_MAPPING_DICT = {1: "A", 2: "C", 3: "G", 4: "T"}

def DNA2Signal(sequence: str) -> np.array:
    """Returns a signal if array do not contains any Ns"""
    s = np.zeros(len(sequence))
    for i, nt in enumerate(sequence):
        if nt == "N":
            return np.zeros(len(sequence))
        s[i] = MAPPING_DICT[nt]
    return s

def Signal2DNA(signal):
    return "".join([INVERSE_MAPPING_DICT[num] for num in signal])


def display_sequences(sequences: Union[List[str], str]):

    if isinstance(sequences, str):
        sequences = [sequences]

    num_subplots = len(sequences)
    
    # Create subplots
    fig, axs = plt.subplots(num_subplots, 1)

    for i, ax in enumerate(axs):
        signal = DNA2Signal(sequences[i])
        ax.plot(signal)
        ax.set_title(f"Sequence {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("nt")

    plt.tight_layout()
    plt.show()

def viz(x, Tx, Wx):
    """Visualize colormap of a sequence x"""
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()


def get_resnet_for_fine_tuning(num_classes: int) -> resnet50:

    pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Set all requires_grad to True for fine-tuning with 20.000 imgs                               
    for param in pretrained_resnet.parameters():
        param.requires_grad = True

    # Override the fc layer with the number of classes
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)
    
    for param in pretrained_resnet.fc.parameters():
        param.requires_grad = True
    
    # Return the new model
    return pretrained_resnet

def extract_sequences_FASTA(fastafile: str):
    with open(fastafile, 'r') as fopen:
        return [line.upper().strip() for line in fopen if not line.startswith('>')]

def extract_sequences_FASTA_with_coords(fastafile: str):

    out = []
    i = 0

    with open(fastafile, 'r') as fopen:

        for line in fopen:
            if line.startswith('>'):
                start_and_stop = line[1:].split(":")[1].split("-")
                start, stop = int(start_and_stop[0].strip()), int(start_and_stop[1].strip())
                i += 1
            else:
                sequence = line.upper().strip()
                i += 1
            if i % 2 == 0:
                out.append([sequence, start, stop])

        return out

def extract_sequences_hf(seqs: List[str], labs: List[int], lab_value: int) -> List[str]:
    seqs = np.array(seqs)
    labs = np.array(labs)

    # Select only sequences belonging to lab_value
    seqs = seqs[np.where(labs == lab_value)[0]]

    return seqs


def make_weigths_for_balanced_classes(images, nclasses):
    """ Returns an array indicating for each image its weight to be sampled """
    count = [0] * nclasses

    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def extract_consecutive_windows(arr):
    """Extract all windows of consecutive integers from an array."""
    if len(arr) == 0:
        return []
    
    # Find where consecutive differences != 1
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    # Split array at break points
    windows = np.split(arr, breaks)
    
    return [window.tolist() for window in windows]

def get_longest_window(windows):
    """Return the longest window from a list of windows."""
    if len(windows) == 1:
        return windows[0]
    return max(windows, key=len)


