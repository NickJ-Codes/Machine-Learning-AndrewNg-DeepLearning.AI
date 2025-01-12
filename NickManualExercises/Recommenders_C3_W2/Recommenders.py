import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from pathlib import Path
import os
from recsys_utils import *

pd.set_option("display.precision", 1)


current_file = Path(__file__)
project_root = current_file.parent.parent.parent
# image_path = project_root / 'C3_UnsupervisedLearning' / 'W1' / 'Lab' / 'bird_small.png' # original image
print(project_root)