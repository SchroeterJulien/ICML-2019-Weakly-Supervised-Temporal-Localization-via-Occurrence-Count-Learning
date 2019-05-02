# Script performing the final inference and evaluation of the model
# The labels are directly extracted from original label files.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import Display.localizationPlot as lp
from new_createDataset import *

# Inference reference name
infer_run_name = "_xtra_"

# Load train-test split
_, test_files = newSplitDataset()

# Initialize results dictionary
scores = {'f1': [], 'precision': [], 'recall': []}

# Compute final score for each sound extract in the test set
for files in test_files:
    print(files.split("/")[-1])

    # Load inference
    prediction_list = np.load('infer/' + infer_run_name + "_" + files.split("/")[-1].replace('.wav', '.npy'))

    # Load labels
    label_tmp = np.loadtxt(files.replace('.wav', '.txt'), skiprows=1)
    label_raw = label_tmp[:, [0, 2]]

    # Compute performance measures
    _, stats = lp.localizationPlotList([prediction_list], [label_raw], decimals=7, bias=-0.030, n_samples=1,
                                           hit_color=["blue", "grey", "red"])

    # Save performance measure
    scores['f1'].append(stats['f1'])
    scores['precision'].append(stats['precision'])
    scores['recall'].append(stats['recall'])

# Compute final score as mean of scores over extracts
print("---------")
print(np.mean(scores['f1']), np.mean(scores['precision']), np.mean(scores['recall']))
print("---------")

# 0.7199368469991827 0.762246590696111 0.6860680323416678



