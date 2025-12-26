import os
import os.path as osp
import random
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)