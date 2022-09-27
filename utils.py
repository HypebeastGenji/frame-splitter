###
### UTILS.PY
###

import h5py

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