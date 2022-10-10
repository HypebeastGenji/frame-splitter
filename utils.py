###
### UTILS.PY
###

import h5py
import numpy as np
import matplotlib.pyplot as plt

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