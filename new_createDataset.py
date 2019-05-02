# File containing all functions for the SMT-Drums dataset creation

import librosa
from multiprocessing import Pool
import numpy as np
import os
import os.path
from sklearn.externals.joblib.parallel import parallel_backend
import soundfile as sf

from google_create_dataset import *

from AudioProcessing.mel_spectrogram import *

# General settings
settings = {}
settings['sample_rate'] = 44100

# Spectrogram setting # todo: pass this through config
spectrum_settings = {}
spectrum_settings['frame_size'] = 0.025
spectrum_settings['frame_stride'] = 0.005  # 0.01
spectrum_settings['NFFT'] = 4096 * 2


def newSplitDataset():
    # Obtain splits from Google code

    test_ids, test_samples = generate_test_set()
    train_samples = generate_train_set(test_ids)

    return train_samples, test_samples


def processSplitSample(input):
    # Function processes a single sound extract
    # Programmed such as this can be done in a parallel manner

    # Process input
    file = input[0]
    infer = input[1]
    config = input[2]
    n_channel = 120

    # Number of filters
    spectrum_settings['number_filter'] = config['n_filter'] // 2

    # Different configuration depending on training or inference
    split_length = config['split_length']
    if infer:
        split_step = 0.5
    else:
        split_step = split_length

    # Default values
    augmentation_factor = config['augmentation_factor']
    if len(input) == 4:
        augmentation_factor = 1
    else:
        pass

    # Load audio
    signal, sr = sf.read(file)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    assert (sr == settings['sample_rate'])

    # Load annotations
    label_raw = np.loadtxt(file.replace('.wav', '.txt'), skiprows=1)

    # Initialize placeholder for data
    final_data = np.zeros(
        [int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor), config['time_steps'],
         config['n_filter']])
    labels = np.zeros([int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor), config['time_steps'],
                       n_channel], np.int8)
    factor = np.zeros([int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor)])

    # Create data extracts
    idx_sample = 0
    for idx_split in range(int(np.ceil(signal.shape[0] / (sr * split_step)))):
        for idx_augmentation in range(augmentation_factor):
            if len(input) == 4:
                idx_augmentation = input[3]

            flag = True

            # Add optional padding to the sound extract
            extended_signal = np.concatenate([1e-5 * np.random.randn(int(config['signal_pad'] * sr)), signal[int(
                idx_split * sr * split_step):int(sr * (idx_split * split_step + split_length))]])

            if idx_augmentation <= 0:
                # Original extract

                # No stretching
                stretching_factor = 1
                signal_stretch = extended_signal

            elif idx_augmentation >= 4:
                # Combine two extracts (play them simultaneously)

                # No stretching
                stretching_factor = 1

                # Select randomly a second extract for superposition
                idx_tmp = np.random.randint(0, int(np.ceil(signal.shape[0] / (sr * split_step))) - 1)

                # Superpose the two extracts
                signal_stretch = extended_signal
                signal_new = np.concatenate([1e-5 * np.random.randn(int(config['signal_pad'] * sr)), signal[int(
                    idx_tmp * sr * split_step):int(sr * (idx_tmp * split_step + split_length))]])

                if signal_stretch.shape[0] == signal_new.shape[0]:
                    signal_stretch += signal_new
                else:
                    # This occurs at the far end of the extract.
                    flag = False


            else:
                # Otherwise, apply random time stretching
                stretching_factor = 0.75 + np.random.rand() / 4
                signal_stretch = librosa.effects.time_stretch(extended_signal, stretching_factor)

            if flag:

                # Save stretching factor
                factor[idx_sample] = stretching_factor

                # Compute spectrogram
                spectrum = melSpectrogram(signal_stretch + np.ones(signal_stretch.shape[0]) * config['signal_noise'],
                                          sr, spectrum_settings['frame_size'],
                                          spectrum_settings['frame_stride'],
                                          config['spectral_size'], spectrum_settings['NFFT'],
                                          normalized=True)

                spectrum = spectrum[:, :config['n_filter'] // 2]

                # Save spectrogram
                final_data[idx_sample, :spectrum.shape[0], :config['n_filter'] // 2] = spectrum

                # Add first order-derivative
                final_data[idx_sample, 1:spectrum.shape[0], config['n_filter'] // 2:] = np.diff(spectrum, n=1,
                                                                                                axis=0)
                # Compute labels (continuous time into spectrogram bins)
                for ii in range(label_raw.shape[0]):
                    if label_raw[ii, 0] - split_step * idx_split >= 0 and label_raw[ii, 0] - split_step * idx_split < split_length:
                        labels[idx_sample, signal2spectrumTime(
                            (np.round(label_raw[ii, 0] - split_step * idx_split, 3) + config['signal_pad']) / stretching_factor *
                            settings['sample_rate']), int(
                            label_raw[ii, 2])] += 1

                # Superpose labels if extracts have been superposed
                if idx_augmentation >= 4:
                    for ii in range(label_raw.shape[0]):
                        if label_raw[ii, 0] - split_step * idx_tmp >= 0 and label_raw[ii, 0] - split_step * idx_tmp < split_length:
                            labels[idx_sample, signal2spectrumTime(
                                (np.round(label_raw[ii, 0] - split_step * idx_tmp, 3) + config['signal_pad']) / stretching_factor *
                                settings['sample_rate']), int(
                                label_raw[ii, 2])] += 1

                idx_sample += 1


    # No double triggering
    labels[labels > 1] = 1

    return final_data.astype(np.float32), labels, factor, np.array([file] * factor.shape[0])


def generateSplitDataset(file_list, config, infer=False):
    # Generate dataset based on list of files
    # Calls the function processSplitSample for parallel processing of samples

    # Initialize final dataset
    x_data_final = np.zeros([0, config['time_steps'], config['n_filter']], np.float32)
    y_data_final = np.zeros([0, config['time_steps'], 120], np.int8)
    factor_final = np.zeros([0])
    file_final = np.zeros([0])

    if len(file_list) > 1:
        print('multiprocessing')
        for ii in range(int(np.ceil(len(file_list) / config['cores']))):
            p = Pool(config['cores'])
            if not infer:
                data_simulated = p.map(processSplitSample, [(x, infer, config, idx_augm) for x in
                                                            file_list[ii * config['cores']:(ii + 1) * config['cores']]
                                                            for idx_augm in np.arange(config['augmentation_factor'])])
            else:
                data_simulated = p.map(processSplitSample, [(x, infer, config) for x in
                                                            file_list[ii * config['cores']:(ii + 1) * config['cores']]])
            p.close()

            # Save data extracts
            x_data = np.concatenate([x[0][:, :, :] for x in data_simulated], axis=0)
            y_data = np.concatenate([x[1][:, :, :] for x in data_simulated], axis=0)
            factor_list = np.concatenate([x[2] for x in data_simulated], axis=0)
            file_tmp = np.concatenate([x[3] for x in data_simulated], axis=0)

            x_data_final = np.concatenate([x_data_final, x_data], axis=0)
            y_data_final = np.concatenate([y_data_final, y_data], axis=0)
            factor_final = np.concatenate([factor_final, factor_list], axis=0)
            file_final = np.concatenate([file_final, file_tmp], axis=0)

            del data_simulated, x_data, y_data, factor_list, file_tmp
    else:
        print('single')
        x_data_final, y_data_final, factor_final, file_final = processSplitSample([file_list[0], infer, config])

    # Final data handling lines
    y_data_final = y_data_final.transpose([0, 2, 1])
    y_data_transformed = y_data_final.reshape([y_data_final.shape[0] * y_data_final.shape[1], y_data_final.shape[2]])

    Y_label = np.zeros([len(np.sum(y_data_transformed, axis=1)), config['max_occurence']])
    Y_label[np.arange(len(np.sum(y_data_transformed, axis=1))), np.sum(y_data_transformed, axis=1)] = 1
    Y_label = Y_label.reshape([y_data_final.shape[0], y_data_final.shape[1], config['max_occurence']])

    return x_data_final.astype(np.float32), Y_label.astype(np.int32), y_data_final.astype(
        np.int32), factor_final, file_final


# Utility functions
def signal2spectrumTime(time):
    """
    Converts signal time (seconds) into spectrogram bin location
    :param time: in seconds
    :return: corresponding spectrogram bin location
    """
    if time <= settings['sample_rate'] * spectrum_settings['frame_size']:
        return int(0)
    else:
        time -= settings['sample_rate'] * spectrum_settings['frame_size']
        return int(1 + time // (settings['sample_rate'] * spectrum_settings['frame_stride']))


def spectrum2signalTime(time):
    """
    Converts spectrogram bin location into signal time (seconds)
    :param time: spectrogram bin location
    :return: corresponding signal time (seconds)
    """
    if time == 0:
        return int(0)
    else:
        time -= 1
        return int(
            settings['sample_rate'] * spectrum_settings['frame_size'] + time * settings['sample_rate'] *
            spectrum_settings['frame_stride'])
