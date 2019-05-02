import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

# Plot inference and compute performance measures
# Based on a array of occurrence (spectrogram time)
def localizationPlot(p_pred, y, n_samples=10, start_sample=0, dist_threshold=5,
                     color_patch=['w', 'cornflowerblue'], bg_color=None, factor=2, bias = 0, decimals = 3):

    # Initialize figure
    fig = plt.figure(figsize=(20, 9))
    count_statistics = np.zeros([3])

    time_steps = y.shape[2]

    for ii in range(p_pred.shape[0]):
        if ii < n_samples:
            ax = plt.subplot(4, np.ceil(n_samples / 4), ii + 1)
        for jj in range(y.shape[1]):
            list_hits = (np.where(y[ii + start_sample, jj, :].astype(np.int16) == 1)[0] // factor)
            list_hits_pred = np.where(p_pred[ii + start_sample, jj, :])[0] + bias
            distance_matrix = np.abs(
                np.tile(list_hits[np.newaxis, :], [len(list_hits_pred), 1]).T - np.tile(list_hits_pred[np.newaxis, :],
                                                                                        [len(list_hits), 1]))

            while distance_matrix.size > 0 and np.min(distance_matrix) <= dist_threshold:
                idx_min = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

                if ii < n_samples:
                    plt.scatter(list_hits[idx_min[0]] / 200 * factor, jj, s=80, facecolors='none',
                                edgecolors=np.reshape([0.4, 0.4, 0.4], [1, 3]))
                    plt.scatter(list_hits_pred[idx_min[1]] / 200 * factor, jj, c='cornflowerblue', marker='*')

                list_hits = np.delete(list_hits, idx_min[0])
                list_hits_pred = np.delete(list_hits_pred, idx_min[1])
                distance_matrix = np.delete(np.delete(distance_matrix, idx_min[0], axis=0), idx_min[1], axis=1)

                count_statistics[0] += 1

            if ii < n_samples:
                plt.scatter(list_hits / 200 * factor, [jj] * len(list_hits), s=80, facecolors='none',
                            edgecolors='crimson')

                plt.scatter(list_hits_pred / 200 * factor, [jj] * len(list_hits_pred), c='crimson', marker='*')

            count_statistics[1] += len(list_hits)
            count_statistics[2] += len(list_hits_pred)

            if ii < n_samples:
                r2 = patches.Rectangle((-0.3, jj - 0.5), 8, 1)
                collection = PatchCollection([r2], facecolor=color_patch[jj % 2], alpha=0.05)
                ax.add_collection(collection)

                if bg_color is  not None and bg_color[ii + start_sample] == 1:
                    ax.set_axis_bgcolor([0.9, 0.9, 1])

        if ii < n_samples:
            plt.xlim([-0.3, (time_steps + 2) / 200])
            # plt.ylim([-0.5, y.shape[1] - 1 + 0.5])
            plt.yticks(np.arange(0), (''))
            plt.ylim([-0.5, p_pred.shape[1] - 0.5])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


            plt.xlabel('Sample ' + str(ii), fontweight='bold')

    # Compute measures
    stats = {}
    TP = count_statistics[0]
    FN = count_statistics[1]
    FP = count_statistics[2]

    precision = nonZeroDivision(TP, TP + FP)
    recall = nonZeroDivision(TP, TP + FN)
    f1 = 2 * nonZeroDivision(precision * recall, precision + recall)

    # Display measures
    print(np.round(f1, decimals), np.round(precision, decimals), np.round(recall, decimals))

    # Save measures
    stats['f1'] = f1
    stats['precision'] = precision
    stats['recall'] = recall

    # Display measures on plot
    plt.text((time_steps + 2) / 100 * 0.5, -0.3,
             str(int(np.round(100 * precision,0))) + '%/' + str(int(np.round(100 * recall, 0))) + '%',fontweight='bold')

    return fig, stats

# Plot inference and compute performance measures
# Based on a list of occurrence (signal time)
def localizationPlotList(pp_list, yy_list, n_samples=10, start_sample=0, dist_threshold=0.050,
                            color_patch=['w', 'cornflowerblue'], bg_color=None, decimals = 3, bias = 0,
                            hit_color = ['cornflowerblue','crimson', np.reshape([0.4, 0.4, 0.4], [1, 3])]):
    # Initialize figure
    fig = plt.figure(figsize=(20, 9))
    count_statistics = np.zeros([3])

    for ii in range(len(pp_list)):
        if ii < n_samples:
            ax = plt.subplot(np.ceil(np.sqrt(n_samples)), np.ceil(n_samples / np.ceil(np.sqrt(n_samples))), ii + 1)
        for jj in range(np.max([np.max(x[:,1]) for x in yy_list]).astype(np.int)+1):
            list_hits = yy_list[ii][yy_list[ii][:,1]==jj,0]
            list_hits_pred = pp_list[ii][pp_list[ii][:,1]==jj,0] + bias
            distance_matrix = np.abs(
                np.tile(list_hits[np.newaxis, :], [len(list_hits_pred), 1]).T - np.tile(list_hits_pred[np.newaxis, :],
                                                                                        [len(list_hits), 1]))

            while distance_matrix.size > 0 and np.min(distance_matrix) <= dist_threshold:
                idx_min = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

                if ii < n_samples:
                    plt.scatter(list_hits[idx_min[0]], jj, s=80, facecolors='none', edgecolors=hit_color[2])
                    plt.scatter(list_hits_pred[idx_min[1]], jj, c=hit_color[0], marker='*')

                list_hits = np.delete(list_hits, idx_min[0])
                list_hits_pred = np.delete(list_hits_pred, idx_min[1])
                distance_matrix = np.delete(np.delete(distance_matrix, idx_min[0], axis=0), idx_min[1], axis=1)

                count_statistics[0] += 1

            if ii < n_samples:
                plt.scatter(list_hits, [jj] * len(list_hits), s=80, facecolors='none',
                            edgecolors=hit_color[1])

                plt.scatter(list_hits_pred, [jj] * len(list_hits_pred), c=hit_color[1], marker='*')

            count_statistics[1] += len(list_hits)
            count_statistics[2] += len(list_hits_pred)

            if ii < n_samples:
                r2 = patches.Rectangle((-0.3, jj - 0.5), 1000, 1)
                collection = PatchCollection([r2], facecolor=color_patch[jj % 2], alpha=0.05)
                ax.add_collection(collection)

                if bg_color is  not None and bg_color[ii + start_sample] == 1:
                    ax.set_axis_bgcolor([0.9, 0.9, 1])

        if ii < n_samples:
            plt.xlim([-0.3, (np.max([np.max(x[:,0]) for x in yy_list]) + 2)])
            # plt.ylim([-0.5, y.shape[1] - 1 + 0.5])
            plt.yticks(np.arange(0), (''))
            plt.ylim([-0.5, np.max([np.max(x[:,1]) for x in yy_list]).astype(np.int) + 1 - 0.5])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


            plt.xlabel('Sample ' + str(ii), fontweight='bold')

    # Compute measures
    stats = {}
    TP = count_statistics[0]
    FN = count_statistics[1]
    FP = count_statistics[2]

    precision = nonZeroDivision(TP, TP + FP)
    recall = nonZeroDivision(TP, TP + FN)
    f1 = 2 * nonZeroDivision(precision * recall , precision + recall)

    # Display measures on plot
    plt.text((np.max([np.max(x[:,0]) for x in yy_list]) + 2) / 100 * 0.5, -0.3,
                 str(int(np.round(100 * f1,0))) + '%  '
               + str(int(np.round(100 * precision, 0))) + '%  '
               + str(int(np.round(100 * recall, 0))) + '%',
                 fontweight='bold', fontsize = 15)

    # Save measures
    stats['f1'] = f1
    stats['precision'] = precision
    stats['recall'] = recall

    # Display measures
    print(np.round(f1, decimals), np.round(precision, decimals), np.round(recall, decimals))

    return fig, stats

def nonZeroDivision(x,y):
    if y==0:
        return 0
    else:
        return x/y
