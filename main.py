import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from config import load_configurations
import Display.localizationPlot as lp
from new_createDataset import *
from weaklyNetworkModel import WeaklyNetwork

# Load configurations
config = load_configurations()
print("---", config['extension'], "---")

# Set visible GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU_Device']

# Open tensorflow session
with tf.Session() as sess:
    # Initialize model
    softModel = WeaklyNetwork(config)
    softModel.initialize(sess)

    # Load dataset
    print('>> Load Dataset...')
    train_files, test_files = newSplitDataset()
    print(len(train_files), len(test_files))

    # Generate training set
    X_data, Y_label, Y_data_raw, _, _ = generateSplitDataset(
        np.random.choice(train_files, config['dataset_size'], replace=False), config)

    pad = config['start_channel']
    Y_label = Y_label[:, pad:pad + config['n_channel'], :]
    Y_data_raw = Y_data_raw[:, pad:pad + config['n_channel'], :]

    # Generate test set
    x_out, y_out_label, y_out_raw, stretch_factor_out, _ = generateSplitDataset(test_files[:3], config)
    y_out_label = y_out_label[:, pad:pad + config['n_channel'], :]
    y_out_raw = y_out_raw[:, pad:pad + config['n_channel'], :]
    print(np.sum(Y_data_raw), np.sum(y_out_raw))

    # Generate validation set (here simply use the first extracts of test set, since no parameter tuning is done)
    x_val = x_out[0:32, :, :]
    y_val_label = y_out_label[0:32, :, :]
    y_val_raw = y_out_raw[0:32, :, :]


    # Inilialize dictionary to save training measures
    print('>>> Training:')
    stats_history = {'f1': [], 'precision': [], 'recall': [],
                     'f1_val': [], 'precision_val': [], 'recall_val': []}

    iter = 1
    while iter <= config['niter']:

        # Select random batch
        idx_batch = np.random.randint(0, len(Y_label), config['batch_size'])
        batch_x, batch_y, batch_y_series = \
            X_data[idx_batch, :, :], Y_label[idx_batch, :, :], Y_data_raw[idx_batch, :, :]

        # Run backpropagation
        softModel.optimize(sess, batch_x, batch_y, batch_y_series)

        # Update the dataset with new extracts (allow to work with lower memory requirement)
        if iter % config['dataset_update_frequency'] == 0 and iter > 0:

            # Generate new data
            print('Dataset Update...')
            x_new, y_new, y_new_raw, _, _ = generateSplitDataset(
                np.random.choice(train_files, config['dataset_update_size'], replace=False), config)
            y_new = y_new[:, pad:pad + config['n_channel'], :]
            y_new_raw = y_new_raw[:, pad:pad + config['n_channel'], :]

            idx_new = np.random.choice(X_data.shape[0], x_new.shape[0], replace=False)

            # Update training dataset
            X_data[idx_new, :, :] = x_new
            Y_label[idx_new, :] = y_new
            Y_data_raw[idx_new, :, :] = y_new_raw

        # Show training status
        if iter % config['show_frequency'] == 0:

            # Performance and predictions of batch data
            acc, los, pp = softModel.infer(sess, batch_x, batch_y, batch_y_series)

            # Display results
            print("For iter ", iter)
            print("Accuracy ", acc)
            if config['Direct']:
                print("Loss ", np.round(los, 3))
            else:
                print("Loss ", np.round(los, 3))
            print("__________________")

            # Display and assess (train) localization
            fig, stats = lp.localizationPlot(
                pp,
                batch_y_series, n_samples=20, dist_threshold=config['tolerence'], factor=config['downsamplingFactor'],
                bias=config['temporal_bias'])
            plt.savefig('plt/localization_in_' + config['extension'])
            plt.close()

            # Save performance
            stats_history['f1'].append(stats['f1'])
            stats_history['precision'].append(stats['precision'])
            stats_history['recall'].append(stats['recall'])

            # Display and assess (validation) localization
            pp = softModel.predict(sess, x_val)
            fig, stats_out = lp.localizationPlot(pp, y_val_raw, n_samples=20, dist_threshold=config['tolerence'],
                                                 factor=config['downsamplingFactor'], bias=config['temporal_bias'])
            plt.savefig('plt/localization_out_' + config['extension'])
            plt.close()

            # Save performance
            stats_history['f1_val'].append(stats_out['f1'])
            stats_history['precision_val'].append(stats_out['precision'])
            stats_history['recall_val'].append(stats_out['recall'])

            # Display Loss & Performance
            softModel.performancePlot(stats_history)

        iter += 1
