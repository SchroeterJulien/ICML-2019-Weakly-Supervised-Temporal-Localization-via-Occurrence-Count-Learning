# Script performing the inference and evaluation of smt-models

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import os.path

from config import load_configurations
import Display.localizationPlot as lp
from Math.NadarayaWatson import *
from new_createDataset import *
from weaklyNetworkModel import WeaklyNetwork

# Inference reference name
infer_run_name = "_xtra_"

# List of models
model_list = ["xtra_25", "xtra_35", "xtra_45", "xtra_55", "xtra_65", "xtra_75", "xtra_85", "xtra_95"]

# Load train-test split
_, test_files = newSplitDataset()

# For each test file compute predictions and scores
for idx_file in range(len(test_files)):

    infer_files = test_files[idx_file:idx_file + 1]
    print("----", infer_files[0], "----")

    # Check if inference has not already been done
    if not os.path.isfile('infer/' + infer_run_name + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy')):

        #
        prediction_list = np.zeros([0, 2])
        for model in model_list:

            # Load configuration
            config = load_configurations(model)

            # Overwrite infrastructure settings
            config['augmentation_factor'] = 4
            config['cores'] = 8

            # Inference settings
            ensembling_factor = 0.02 * config['augmentation_factor']
            suppression_field = 20
            threshold = 0.35

            print("---", config['extension'], "---")

            with tf.Session() as sess:
                # Restore model
                softModel = WeaklyNetwork(config)
                softModel.restore(sess)

                # Load out-of-sample data
                print('>> Load Dataset...')
                x_out, _, y_out_raw, stretch_factor_out, file_list_out = generateSplitDataset(infer_files, config,
                                                                                              infer=True)

                pad = config['start_channel']
                y_out_raw = y_out_raw[:, pad:pad + config['n_channel'], :]

                # Single extract score
                pp = softModel.predict(sess, x_out)
                _, _ = lp.localizationPlot(pp, y_out_raw, n_samples=20, dist_threshold=config['tolerence'], factor=1,
                                           bias=config['temporal_bias'], decimals=7)
                plt.close()

                sess.close()
            softModel.reset()


            # Ensembling of data augmented extracts
            pp_trans = np.transpose(pp.reshape(
                [pp.shape[0] // config['augmentation_factor'], config['augmentation_factor'], pp.shape[1], pp.shape[2]]),
                [1, 0, 2, 3])
            pp_ensemble = softModel.ensembling(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)

            # Data augmentation score
            _, _ = lp.localizationPlot(pp_ensemble, y_out_raw[::config['augmentation_factor'], :, :], n_samples=10,
                                       dist_threshold=config['tolerence'],
                                       factor=1, bias=config['temporal_bias'], decimals=7)



            # Reassemble overlapping samples
            _start_extract = 0

            y_ensemble = y_out_raw[::config['augmentation_factor'], :, :]
            file_list_out_ensemble = file_list_out[::config['augmentation_factor']]
            y_pasted = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
            pp_pasted = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
            ww = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
            file_out_unique = []
            previous_source = ""
            idx_source = -1
            for ii in range(len(file_list_out_ensemble)):
                if file_list_out_ensemble[ii] == previous_source:
                    idx_start += 100  # 0.5 step
                else:
                    idx_start = 0
                    idx_source += 1
                    previous_source = file_list_out_ensemble[ii]
                    file_out_unique.append(previous_source)

                y_pasted[idx_source, :, idx_start:idx_start + y_ensemble[ii, :, _start_extract:].shape[1]] += y_ensemble[ii,
                                                                                                              :,
                                                                                                              _start_extract:]
                pp_pasted[idx_source, :, idx_start:idx_start + pp_ensemble[ii, :, _start_extract:].shape[1]] += pp_ensemble[
                                                                                                                ii,
                                                                                                                :,
                                                                                                                _start_extract:]
                ww[idx_source, :,
                idx_start:idx_start + pp_ensemble[ii, :, _start_extract:300 + _start_extract].shape[1]] += 1

            # Normalize
            pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
            y_final = y_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] > 0

            # Load labels from file
            yy = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
            yy_list = []
            for jj in range(yy.shape[0]):
                label_tmp = np.loadtxt(file_out_unique[jj].replace('.wav', '.txt'), skiprows=1)
                label_raw = label_tmp[:, [0, 2]]
                label_raw = label_raw[[x >= pad and x < pad + config['n_channel'] for x in label_raw[:, 1]], :]
                label_raw[:, 1] -= pad
                for kk in range(label_raw.shape[0]):
                    yy[jj, int(label_raw[kk, 1]), int(label_raw[kk, 0] * 200)] += 1

                yy_list.append(label_raw)


            # Final prediction cleaning
            if True:
                pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
                pp_final_cleaning = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
                for ii in range(pp_final_cleaning.shape[0]):
                    for jj in range(pp_final_cleaning.shape[1]):
                        for tt in range(pp_final_cleaning.shape[2]):
                            if pp_final[ii, jj, tt] > 0:
                                if np.sum(pp_final[ii, jj, tt:tt + suppression_field]) >= threshold:
                                    pp_final_cleaning[ii, jj, tt] = 1
                                    pp_final[ii, jj, tt:tt + suppression_field] = 0
            else:
                # New final cleaning (Not used since to slow)
                # todo: speed up this!
                pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
                mass_threshold = 0.05 * threshold
                pp_final_cleaning = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
                sample_idx = 0
                for note_idx in range(pp_final_cleaning.shape[1]):
                    flag = 0
                    while flag == 0:
                        full = GaussKernel(np.arange(pp_final.shape[2]), np.arange(pp_final.shape[2]),
                                           pp_final[sample_idx, note_idx, :], 5)

                        if np.max(full) > mass_threshold:
                            idx_max = np.where(full == np.max(full))[0][0]
                            local_idx_max = int(idx_max)

                            pp_final[sample_idx, note_idx,
                                max(local_idx_max - suppression_field, 0):local_idx_max + suppression_field] = 0
                            pp_final_cleaning[sample_idx, note_idx, local_idx_max] += 1

                            print(idx_max)
                        else:
                            flag = 1

            # Assess the performance of reassemble predictions
            fig, _ = lp.localizationPlot(pp_final_cleaning[:, :, :], yy[:, :, :], n_samples=pp_final_cleaning.shape[0],
                                         dist_threshold=config['tolerence'],
                                         factor=1, bias=config['temporal_bias'], decimals=7)

            # Compute prediction time in seconds (not number of extracts anymore)
            pp_list = []
            for ii in range(pp_final.shape[0]):
                triggers = np.zeros([0, 2])
                for jj in range(pp_final.shape[1]):
                    list_hits = np.where(pp_final_cleaning[ii, jj])[0] / 200
                    triggers = np.concatenate([triggers, np.concatenate(
                        [list_hits[:, np.newaxis], np.array([jj] * len(list_hits))[:, np.newaxis]], axis=1)])
                pp_list.append(triggers)

            try:
                fig, _ = lp.localizationPlotList(pp_list, yy_list, decimals=7, bias=-0.030, n_samples=1)
            except:
                # The function might crash if the number of predictions is zero, so simply  pass it.
                pass
            plt.close('all')

            print("---", config['extension'], "---")

            # Save predictions for this range of 10 notes
            tmp_list = pp_list[0]
            tmp_list[:, 1] += config['start_channel']
            prediction_list = np.concatenate([prediction_list, tmp_list], axis=0)

        # Final performance computation
        # Load the original labels directly from files
        label_tmp = np.loadtxt(infer_files[0].replace('.wav', '.txt'), skiprows=1)
        label_raw = label_tmp[:, [0, 2]]

        # Compute performance measures and save result
        fig, _ = lp.localizationPlotList([prediction_list], [label_raw], decimals=7, bias=-0.030, n_samples=1)
        # The temporal bias takes into consideration the bias induced when the signal time is transformed to
        # spectrogram time and back to signal time without any correction.
        plt.savefig('plt/inference/' + infer_run_name + "_" + infer_files[0].split("/")[-1].replace('.wav', '.png'))
        plt.close('all')

        # Save predictions for further inference
        np.save('infer/' + infer_run_name + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy'), prediction_list)

    else:
        print("<<< Already exists!")
