import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import scipy
import tensorflow as tf
from tensorflow.contrib import rnn
from multiprocessing import Pool

import Display.localizationPlot as lp
from Math.NadarayaWatson import *


class WeaklyNetwork:

    def __init__(self, configuration):
        print('>> Construct Graph...')
        os.environ["CUDA_VISIBLE_DEVICES"] = configuration['GPU_Device']

        self.config = configuration

        # Loss placeholders
        self.loss_window = {'loss': np.zeros([25])}
        self.list_loss = {'loss': []}

        # Iteration count
        self.iter = 0

        # Output layer weights and biases
        hidden_weights = tf.Variable(tf.random_normal([self.config['num_units'], self.config['hidden_size']]))
        hidden_bias = tf.Variable(tf.zeros([self.config['hidden_size']]))
        out_weights = tf.Variable(tf.random_normal([self.config['hidden_size'], self.config['n_channel']]))
        out_bias = tf.Variable(
            np.log((1 - np.power(0.8, 1 / self.config['time_steps'])) / np.power(0.8, 1 / self.config[
                'time_steps'])) * tf.ones([self.config['n_channel']]))  # Y[0,time_step] = 0.8

        # Placeholder
        self.x = tf.placeholder("float", [None, self.config['time_steps'], self.config['n_filter']])
        self.y = tf.placeholder("float", [None, self.config['n_channel'], self.config['max_occurence']])
        self.y_series = tf.placeholder("float", [None, self.config['n_channel'], self.config['time_steps']])

        # Downsampling spectrogram
        x_conv = tf.expand_dims(self.x, dim=3)  # l x 144
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][0], kernel_size=[3, 4])  # 384
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 3], strides=[1, 3])  # l x 72
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][1], kernel_size=[3, 4])  # 128
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 36
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][2], kernel_size=[3, 4])  # 64
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 18
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][3], kernel_size=[3, 4])  # 32
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[self.config['downsamplingFactor'], 2],
                                         strides=[self.config['downsamplingFactor'], 2])  # l x 9

        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][4], kernel_size=[3, 4])  # 16
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 3
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][5], kernel_size=[3, 4])  # 8
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 1
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][6], kernel_size=[3, 4])  # 4
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 1
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][7], kernel_size=[3, 4])  # 2
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 1
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][8], kernel_size=[3, 4])  # 2
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 1

        # processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
        x_conv = tf.reshape(x_conv,
                            [self.config['batch_size'], self.config['time_steps'] // self.config['downsamplingFactor'],
                             self.config['n_filters'][8]])

        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length

        # Reccuring unit
        length_sequence = length(self.x) // self.config['downsamplingFactor'] - 2
        lstm_layer = rnn.BasicLSTMCell(self.config['num_units'], forget_bias=1)
        output_rnn, _ = tf.nn.dynamic_rnn(lstm_layer, x_conv, dtype=tf.float32, sequence_length=length_sequence)

        def last_relevant(output, length):
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (length - 1)
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant

        prediction = tf.one_hot(tf.zeros(self.config['batch_size'] * self.config['n_channel'], dtype=tf.int32),
                                self.config['max_occurence'])  # initial prediction
        output_rnn_stack = tf.unstack(output_rnn, self.config['time_steps'] // self.config['downsamplingFactor'], 1)

        outputs = []
        pp_list = []
        for output in output_rnn_stack:
            hidden_layer = tf.sigmoid(tf.matmul(output, hidden_weights) + hidden_bias)
            increment = tf.sigmoid(tf.matmul(hidden_layer, out_weights) + out_bias)
            increment = tf.reshape(increment, [self.config['batch_size'] * self.config['n_channel'], 1])

            # todo: use a convolution instead of this mess with [1-alpha, alpha] as filter
            # stayed + moved
            prediction = tf.multiply(
                tf.concat((tf.tile(1 - increment, [1, self.config['max_occurence'] - 1]),
                           tf.ones([self.config['batch_size'] * self.config['n_channel'], 1])), axis=1),
                prediction) + \
                         tf.multiply(tf.tile(increment, [1, self.config['max_occurence']]),
                                     tf.slice(tf.concat(
                                         (tf.zeros([self.config['batch_size'] * self.config['n_channel'], 1]),
                                          prediction),
                                         axis=1), [0, 0],
                                         [self.config['batch_size'] * self.config['n_channel'],
                                          self.config['max_occurence']]))

            outputs.append(prediction)
            pp_list.append(increment)

        # Loss Computation
        prediction = last_relevant(tf.stack(outputs, 1),
                                   tf.reshape(
                                       tf.tile(tf.expand_dims(length_sequence, 1), [1, self.config['n_channel']]),
                                       [self.config['batch_size'] * self.config['n_channel']]))
        y_reshaped = tf.identity(self.y)
        y_reshaped = tf.reshape(y_reshaped,
                                [self.config['batch_size'] * self.config['n_channel'], self.config['max_occurence']])
        self.loss = tf.reduce_mean(
            -tf.reduce_sum(y_reshaped * tf.log(prediction + 1e-9), reduction_indices=[1]))  # does not work

        # optimization
        if self.config['bool_gradient_clipping']:
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.norm = tf.global_norm(grads)
            grads, _ = tf.clip_by_global_norm(grads, self.config['clipping_ratio'])  # gradient clipping
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.opt = optimizer.apply_gradients(grads_and_vars)
        else:
            self.opt = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.loss)

        # model evaluation
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_reshaped, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.stacked_prediction = tf.stack(outputs, axis=1)

        # Model saver
        self.saver = tf.train.Saver()

        # initialize variables
        self.init = tf.global_variables_initializer()

    def reset(self):
        tf.reset_default_graph()

    # One step optimization
    def optimize(self, session, batch_x, batch_y, batch_y_series):

        # Run optimization
        ll, _ = session.run([self.loss, self.opt],
                            feed_dict={self.x: batch_x.astype(np.float),
                                       self.y: batch_y.astype(np.float),
                                       self.y_series: batch_y_series.astype(np.float)})

        self.iter += 1

        if self.iter % 10 == 0:
            # Save window loss
            self.loss_window['loss'][1:] = self.loss_window['loss'][:-1]
            self.loss_window['loss'][0] = ll
            self.list_loss['loss'].append(np.median(self.loss_window['loss'][self.loss_window['loss'] != 0]))

        if self.iter % self.config['save_frequency'] == 0:
            self.save(session)

    # initialize variables
    def initialize(self, session):
        session.run(self.init)

    def save(self, session):
        self.saver.save(session, "models/model" + self.config['extension'] + ".ckpt")
        print("Model saved")

    def restore(self, session):
        self.saver.restore(session, "models/model" + self.config['extension'] + ".ckpt")
        # print("Model Restored!")

    def predict(self, session, xx):
        pp = np.zeros(
            [xx.shape[0], self.config['n_channel'], self.config['time_steps'] // self.config['downsamplingFactor']])

        for ii in range(xx.shape[0] // self.config['batch_size']):
            pred_ts = session.run(self.stacked_prediction, feed_dict={
                self.x: xx[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size']]})
            predictions = np.array([pred_ts])

            pp[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size'], :, :-1] = (
                    1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config[
                'trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel'],
                 self.config['time_steps'] // self.config['downsamplingFactor'] - 1])
            pp[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size'], :, 0] = (
                    1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel']])

        if xx.shape[0] % self.config['batch_size'] > 0:
            pred_ts = session.run(self.stacked_prediction, feed_dict={self.x: xx[-self.config['batch_size']:]})
            predictions = np.array([pred_ts])

            pp[-self.config['batch_size']:, :, :-1] = (
                    1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config[
                'trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel'],
                 self.config['time_steps'] // self.config['downsamplingFactor'] - 1])
            pp[-self.config['batch_size']:, :, 0] = (
                    1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel']])

        return pp

    def evaluate(self, session, x, y):
        pp = self.predict(session, x)

        fig, stats = lp.localizationPlot(
            pp, y, n_samples=20, dist_threshold=self.config['tolerence'], factor=1, bias=self.config['temporal_bias'])

        return fig, stats

    def infer(self, session, batch_x, batch_y, batch_y_series):
        a, d, pred_ts = session.run(
            [self.accuracy, self.loss, self.stacked_prediction],
            feed_dict={self.x: batch_x, self.y: batch_y, self.y_series: batch_y_series.astype(np.float)})

        predictions = np.array([pred_ts])
        pp = np.zeros(
            [batch_y.shape[0], batch_y_series.shape[1], batch_y_series.shape[2] // self.config['downsamplingFactor']])
        pp[:, :, :-1] = (
                1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config['trigger_threshold']).reshape(
            [batch_y.shape[0], batch_y_series.shape[1],
             batch_y_series.shape[2] // self.config['downsamplingFactor'] - 1])
        pp[:, :, 0] = (1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
            [batch_y.shape[0], batch_y_series.shape[1]])

        return a, d, pp

    def performancePlot(self, stats_history):

        f = plt.figure(figsize=(15, 7))
        ax = plt.subplot(1, 3, 1)

        plt.plot(10 * np.arange(1, 1 + len(self.list_loss['loss'])),
                 np.log(np.array(self.list_loss['loss'])))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(True)

        plt.ylim([np.min(np.log(np.array(self.list_loss['loss']))) - 0.02, 0])

        # Train set subplot
        ax = plt.subplot(1, 3, 2)
        standardPerformanceSubplot(ax, self.config['show_frequency'], stats_history['f1'],
                                   stats_history['precision'], stats_history['recall'])

        # Validation set subplot
        ax = plt.subplot(1, 3, 3)
        standardPerformanceSubplot(ax, self.config['show_frequency'], stats_history['f1_val'],
                                   stats_history['precision_val'], stats_history['recall_val'])

        plt.savefig('plt/loss_history_' + self.config['extension'])
        plt.close()

    def ensembling(self, pp, factor, mass_threshold=0.2, suppression_field=14):
        """
        pp: [fold, samples, channel, time_steps]
        :param pp:
        :return:
        """

        pp_ensembling = np.zeros([pp.shape[1], pp.shape[2], pp.shape[3]])
        for sample_idx in range(pp.shape[1]):
            for note_idx in range(pp.shape[2]):

                detected_events = []
                flag = 0
                iteration = 0
                while flag == 0 and iteration < 1000:

                    # plt.figure()
                    full = 0
                    for kk in range(pp.shape[0]):
                        series = pp[kk, sample_idx, note_idx, :]
                        series_conv = GaussKernel(np.arange(pp.shape[3]),
                                                  np.arange(pp.shape[3]) * factor[sample_idx * pp.shape[0] + kk],
                                                  series, 5)
                        full += series_conv

                    if np.max(full) > mass_threshold:
                        idx_max = np.where(full == np.max(full))[0][0]

                        # delete max mass around that point
                        for kk in range(pp.shape[0]):
                            local_idx_max = int(idx_max / factor[sample_idx * pp.shape[0] + kk])
                            pp[kk, sample_idx, note_idx,
                            max(local_idx_max - suppression_field, 0):local_idx_max + suppression_field] = 0

                        detected_events.append(idx_max)
                        pp_ensembling[sample_idx, note_idx, idx_max] += 1
                    else:
                        flag = 1

                    iteration += 1

        return pp_ensembling

    def ensemblingParallel(self, pp, factor, mass_threshold=0.2, suppression_field=14):
        """
        pp: [fold, samples, channel, time_steps]
        :param pp:
        :return:
        """
        p = Pool(self.config['cores'])
        ensembling_output = p.map(processEnsembling,
                                  [(idx, pp, factor, mass_threshold, suppression_field) for idx in range(pp.shape[1])])
        p.close()

        return sum(ensembling_output)


def processEnsembling(arg):
    sample_idx = arg[0]
    pp = arg[1]
    factor = arg[2]
    mass_threshold = arg[3]
    suppression_field = arg[4]

    pp_ensembling = np.zeros([pp.shape[1], pp.shape[2], pp.shape[3]])
    for note_idx in range(pp.shape[2]):

        detected_events = []
        flag = 0
        iteration = 0
        while flag == 0 and iteration < 1000:

            # plt.figure()
            full = 0
            for kk in range(pp.shape[0]):
                series = pp[kk, sample_idx, note_idx, :]
                series_conv = GaussKernel(np.arange(pp.shape[3]),
                                          np.arange(pp.shape[3]) * factor[sample_idx * pp.shape[0] + kk],
                                          series, 5)
                full += series_conv

            if np.max(full) > mass_threshold:
                idx_max = np.where(full == np.max(full))[0][0]

                # delete max mass around that point
                for kk in range(pp.shape[0]):
                    local_idx_max = int(idx_max / factor[sample_idx * pp.shape[0] + kk])
                    pp[kk, sample_idx, note_idx,
                    max(local_idx_max - suppression_field, 0):local_idx_max + suppression_field] = 0

                detected_events.append(idx_max)
                pp_ensembling[sample_idx, note_idx, idx_max] += 1
            else:
                flag = 1

            iteration += 1

    return pp_ensembling


def standardPerformanceSubplot(ax, frequency, f1, precision, recall):
    plt.plot(frequency * np.arange(1, 1 + len(f1)), f1)
    plt.plot(frequency * np.arange(1, 1 + len(precision)), precision)
    plt.plot(frequency * np.arange(1, 1 + len(recall)), recall)
    plt.plot(frequency * np.arange(1, 1 + len(f1)),
             GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5), 'k')

    plt.text(frequency * len(f1), f1[-1], str(np.round(100 * f1[-1], 1)) + '%')

    plt.text(frequency * len(f1),
             GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5)[-1],
             str(np.round(100 * GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5)[-1],
                          1)) + '%')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim([0, 1])
