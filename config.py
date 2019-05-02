# Configuration file
# The load_configurations function allows to set, save and load the configuration used for a specific run.

# Comment:
#    The number of parameters can seem large. However, this does not correspond to the number of free parameters of
#    the approach. Indeed, most parameters are fixed constants and additional optional parameters for further research
#    are included without being used.

import numpy as np
import os.path

# Path to the saving folder
path2models = '/home/icml/maps/models/'


def load_configurations(extension=""):
    """
    sets, saves or loads the parameters for a specific run.
    :param extension: str, name of the run.
    :return: dict, contains all parameters set for the run.
    """

    if os.path.isfile(path2models + extension + '.npy') and extension is not "":
        # Load the configurations if extension already exists
        print('Load from backup')
        config = np.load(path2models + extension + '.npy').item()

    else:
        # Set configurations and save them

        # Initialize dictionary
        config = {}

        # Extension Name
        config['extension'] = 'xtra_25'

        # Channels selected (by block of 10)
        config['start_channel'] = 25

        # Run settings
        config['niter'] = 500000
        config['show_frequency'] = 2500
        config['save_frequency'] = 10000

        # Network settings
        config['num_units'] = 64
        config['hidden_size'] = 32
        config['n_filters'] = [16, 16, 32, 32, 64, 64, 64, 64, 64]
        config['downsamplingFactor'] = 1  # No temporal downsampling

        # Learning settings
        config['clipping_ratio'] = 10
        config['learning_rate'] = 0.0001
        config['batch_size'] = 32
        config['bool_gradient_clipping'] = True

        # Dataset settings
        config['dataset_size'] = 48
        config['dataset_update_size'] = 8
        config['dataset_update_frequency'] = 2500
        config['update_start'] = 0

        # Dataset constants
        config['time_steps'] = 400
        config['n_filter'] = 384 * 2
        config['max_occurence'] = 42
        config['n_channel'] = 10
        config['spectral_size'] = 550

        # Dataset options
        config['augmentation_factor'] = 7
        config['ensembling_factor'] = 0.04 * config['augmentation_factor']
        config['signal_pad'] = 0
        config['signal_noise'] = 0
        config['split_length'] = 1.5

        # Infrastructure
        config['cores'] = 40
        config['GPU_Device'] = "0"

        # Settings for inference
        config['tolerence'] = 10 // config['downsamplingFactor']
        config['temporal_bias'] = 0 // config['downsamplingFactor']
        config['trigger_threshold'] = 0.2

        # Save settings
        np.save(path2models + config['extension'] + '.npy', config)

    return config
