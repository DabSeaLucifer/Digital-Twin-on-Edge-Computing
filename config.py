
"""
implemented FedProx, SGAN components (FeatureDrop, MBD),
Loss-based Client Selection, enhanced client-returned metrics, and server-side loss graphing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import copy

# Hyperparameters
torch.manual_seed(0)
np.random.seed(0) # For client sampling consistency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
FAKE_CLASS_IDX = NUM_CLASSES # Real (0-9), Fake (10)
BATCH_SIZE = 32
Z_DIM = 100
LEARNING_RATE_D = 2e-5
LEARNING_RATE_G = 2e-6
LAMBDA_FM = 1.0 # Feature Matching loss weight
NUM_ROUNDS = 10
NUM_CLIENTS = 4 # Total number of clients available
CLIENTS_PER_ROUND_ACTUAL = 4 # Number of clients to actually select per round
LOCAL_EPOCHS = 2
NUM_LABELED = 100 # Number of labeled samples for the first client
SMOOTHING = 0.1 # Label smoothing
GRADIENT_CLIP_D = 1.0
GRADIENT_CLIP_G = 1.0
GENERATOR_UPDATES_PER_DISCRIMINATOR = 1
MU = 0.01 # FedProx proximal term coefficient

# PDF Component Hyperparameters
FEATURE_DROP_PROB = 0.5 # Probability for Feature Drop layer
MBD_OUT_FEATURES = 50  # Number of output features for MBD (dimensionality of the MBD tensor rows)
