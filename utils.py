###
### UTILS.PY
###

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle

def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file            
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')


def get_session_name(path):

    new_file_name = path[:-19].replace(" ", "-")
    file_start = new_file_name.rfind('/') + 1
    new_file_name = new_file_name[file_start:]

    return new_file_name


def check_dict(split_data):
    keys = ['frames', 'frames_mask', 'metadata', 'scalars', 'timestamps']
    for key in keys:
        for i in split_data[key]:
            if key == 'scalars':
                print(key)
                print(i)
                print(len(split_data[key][i]))
                for j in split_data[key][i]:
                    print(len(j))

            else:
                print(key)
                print(len(i))


def clone_file(old_file):
    with h5py.File('newfile.h5', 'w') as new_h5:
        with h5py.File(old_file, 'r') as old_h5:
            keys = old_h5.keys()
            datasets = []
            for key in keys:
                if isinstance(old_h5[key], h5py.Dataset):
                    new_h5.create_dataset(key, data=old_h5[key]) # change to split data
                    datasets.append(key)
                else:
                    if key not in datasets:
                        print(key)
                        # new_h5.create_group('/'+key)
                        old_h5.copy(old_h5[key], new_h5['/'])
            print('[SUCCESS]: NEW FILE CREATED')


# will populate create empty depth.dat and depth_ts.txt files
def populate_depth(destination, files=['depth.dat', 'depth_ts.txt']):
    for file in files:
        with open(destination + '/' + file, mode='a'): 
            pass


def get_mouse_id(sessions):
    sesh_lists = sessions.split('-')
    if len(sesh_lists) == 3 or len(sesh_lists) == 7:
        mouse_id = sesh_lists[1]
    elif len(sesh_lists) == 5:
        mouse_id = sesh_lists[-2]
    elif len(sesh_lists) == 6:
        for word in sesh_lists:
            if len(word) == 4 or len(word) == 3:
                if word[-1].isnumeric(): # checks last digit instead of whole thing because of WT (e.g WT6 - last character is a number)
                    mouse_id = word
                    
    else:
        print("[ERROR]: error when handling mice id")
        print(len(sessions.split('-')))
        print(sessions)
    
    return mouse_id


def multi_bar_plot(stat_list, session_labels, scalar, title):
    control_means = stat_list[0]
    stim_means = stat_list[1]
    post_means = stat_list[2]

    x = np.arange(len(session_labels))  # label locations
    width = 0.25  # width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, control_means, width, label='Control')
    rects2 = ax.bar(x, stim_means, width, label='Stim')
    rects3 = ax.bar(x + width, post_means, width, label='Post')

    ax.set_ylabel(scalar)
    ax.set_xlabel("Sessions")
    ax.set_title(title)
    ax.set_xticks(x, session_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    # plt.savefig("./WT Analysis/WT "+ scalar +" per frame.png")

    plt.show()


def reorder_list(*args, order):
    ordered_lists = []
    for arg in args:
        ordered_list = [arg[i] for i in order]
        ordered_lists.append(ordered_list)
    return ordered_lists


def get_line(xs, slope, intercept):
    return slope * xs + intercept

def piecewise_linear(data, segments, quiet=False):
    arrays = np.array_split(data, segments)
    regression_array = np.zeros((segments, 2))

    main_regression = stats.linregress(data)
    main_regression_line = get_line(data[:, 0], main_regression.slope, main_regression.intercept)
    if not quiet:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
        plt.plot(data[:, 0], main_regression_line, linewidth=4, color='red')

    for idx, array in enumerate(arrays):
        regression = stats.linregress(array)
        regression_line = get_line(array[:, 0], regression.slope, regression.intercept)

        regression_array[idx, 0] = regression.slope
        regression_array[idx, 1] = regression.intercept

        if not quiet:
            plt.plot(array[:, 0], regression_line, linewidth=4, color='black')
    return regression_array


def new_piecewise_linear(data, segments, quiet=False):
    arrays = np.array_split(data, segments)
    regression_array = np.zeros((segments, 2))

    main_regression = stats.linregress(data)
    main_regression_line = get_line(data[:, 0], main_regression.slope, main_regression.intercept)
    if not quiet:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
        plt.plot(data[:, 0], main_regression_line, linewidth=4, color='red')

    xs = []
    ys = []
    for idx, array in enumerate(arrays):
        regression = stats.linregress(array)
        regression_line = get_line(array[:, 0], regression.slope, regression.intercept)
        median = np.median(array[:, 0])
        mean = np.mean(array[:, 1])
        xs.append(median)
        ys.append(mean)

        regression_array[idx, 0] = regression.slope
        regression_array[idx, 1] = regression.intercept

#         if not quiet:
#             plt.plot(array[:, 0], regression_line, linewidth=4, color='black')
    plt.plot(xs, ys, color='purple', linewidth=4)
    plt.show()
    return regression_array


def plot_comparison(df, groups=None):
    comp_group = ['10Hz WT', '10Hz WT sham']

    conditions = ['control', 'stim', 'post']
    for condition in conditions:
        x_vals = comp_group
        y_vals = [df['10Hz WT'].loc[condition], df['10Hz WT sham'].loc[condition]]
        plt.bar(x_vals, y_vals)
        plt.ylim((0, 3))
        plt.xlabel( '10Hz rTMS (' + str(condition) + ')', fontsize=18)
        plt.ylabel('Velociy (mm/frame)', fontsize=18)
        
        plt.show()


def avg_df(frames, input_type='df'):
    arrays = []
    for df in frames:
        if input_type == 'dict':
            print(df)

            df = pd.DataFrame(df, index=[0])
            print(df)
        arrays.append(df.to_numpy())
    mean = sum(arrays) / 2
    mean_df = pd.DataFrame(mean)
    print(mean_df)
    return mean_df


def pickle_save(data, dest):
    with open(dest, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('[SAVED]: ' + str(dest))

'''
Error bars
converge
anova

'''

''' MIGHT NEED LATER

try:
    del new_h5['/metadata/extraction/flips']
    new_h5.create_dataset('metadata/extraction/flips', data=dataset[i])
except KeyError:
    print("[ERROR]: USING OLD H5 FILE FORMAT")
    cont = input("Would you like to continue anyway? (y/n): ")
    if cont == 'y':
        del new_h5['/metadata/flips']
        new_h5.create_dataset('metadata/flips', data=dataset[i])
    else:
        quit()


try:
    flip_data = data['extraction']['flips']
    # flip_data = data['extraction/flips'] CAN USE '/' to move down hierarchy
except KeyError:
    print("[ERROR]: USING OLD H5 FILE FORMAT")
    cont = input("Would you like to continue anyway? (y/n): ")
    if cont == 'y':
        flip_data = data['flips']
    else:
        quit()

'''