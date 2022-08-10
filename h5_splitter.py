import h5py
import pandas as pd
import numpy as np

# so np arrays dont truncate
import sys
# np.set_printoptions(threshold=sys.maxsize)

filename = "./proc/results_00.h5"


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

## READ WITH H5PY
# h5print(filename)


### READ THROUGH h5 FILE AND EVERY SEPERATE AT 2 CUTTING POINTS (frames)
### def cut_frames(filename, cut_points=[18000, 36000])
### ONCE THESE ARRAYS ARE EDITED SAVE AS A NEW H5 file (3 output)

def split_groups(dataset, cut_points, print_summary=False):
    control_frames = dataset[:cut_points[0]]
    stim_frames = dataset[cut_points[0]:cut_points[1]]
    post_frames = dataset[cut_points[1]:]
    if print_summary == True:
        print("[PRINTING SUMMARY]")
        print(control_frames)
        print(len(control_frames))
        print("--------------------------------------")
        print(stim_frames)
        print(len(stim_frames))
        print("--------------------------------------")
        print(post_frames)
        print(len(post_frames))
        print("--------------------------------------")
    return control_frames, stim_frames, post_frames


def split_h5(filename, cut_points, key=None, print_summary=False):
    with h5py.File(filename, 'r') as f:
        frame_num = f['frames'].shape[0]
        print("Frames in session:", frame_num)
        data_dict = {}
        if key == None:
            keys = f.keys()
        else:
            if key not in f.keys():
                print("NOT A VALID KEY")
                print("Please use an available key in this dataset:", f.keys())
                print("QUITTING")
                quit()
            else:
                keys = [key]
        print("keys:",keys)
        for key in keys:
            print("[SELECTED]:", key.upper())
            data = f[key]
            if isinstance(data, h5py.Dataset):
                print(data)
                control_frames, stim_frames, post_frames = split_groups(data, cut_points, print_summary=print_summary)
                data_dict[key] = [control_frames, stim_frames, post_frames]
                print("[SUCCESS]")
            elif data.name[1:] == 'metadata':
                flip_data = data['extraction']['flips']
                # flip_data = data['extraction/flips'] CAN USE '/' to move down hierarchy
                print(flip_data)
                control_frames, stim_frames, post_frames = split_groups(flip_data, cut_points, print_summary=print_summary)
                data_dict[key] = [control_frames, stim_frames, post_frames]
                print("[SUCCESS]")
            elif data.name[1:] == 'scalars':
                scalars = {}
                for scalar in data:
                    scalar_data = data[scalar]
                    print(scalar_data)
                    control_frames, stim_frames, post_frames = split_groups(scalar_data, cut_points, print_summary=print_summary)
                    scalar_list = [control_frames, stim_frames, post_frames]
                    scalars[scalar] = scalar_list
                    print("[SUCCESS]")
                data_dict[key] = scalars

        return data_dict
# split_data = dictionary of split values    
split_data = split_h5(filename, [18000, 36000], print_summary=False)


def check_dict():
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
# check_dict()

def clone_file():
    with h5py.File('newfile.h5', 'w') as new_h5:
        with h5py.File(filename, 'r') as old_h5:
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
# clone_file()


def create_file():
    new_files = ['control', 'stim', 'post']
    for i in range(len(new_files)):
        print('[CREATING FILE]: ' + new_files[i] + '-newfile.h5')
        with h5py.File(new_files[i] + '-newfile.h5', 'w') as new_h5:
            with h5py.File(filename, 'r') as old_h5:
                keys = old_h5.keys()
                datasets = []
                for key in keys:
                    dataset = split_data[key]
                    if key == 'scalars':
                        for scalar in dataset:
                            scalar_sets = dataset[scalar]
                            new_h5.create_dataset(key+'/'+scalar, data=scalar_sets[i])
                    else:
                        if isinstance(old_h5[key], h5py.Dataset):
                            new_h5.create_dataset(key, data=dataset[i]) 
                            datasets.append(key)
                        else:
                            if key not in datasets:
                                old_h5.copy(old_h5[key], new_h5['/'])
                                if key == 'metadata':
                                    del new_h5['/metadata/extraction/flips']
                                    new_h5.create_dataset('metadata/extraction/flips', data=dataset[i])
                print('[NEW FILE CREATED]:', new_files[i]+'-newfile.h5')
create_file()

# CHECK NEWFILE
h5print('control-newfile.h5')
print()
h5print('stim-newfile.h5')
print()
h5print('post-newfile.h5')



### --------------------------- OLD FUNCTION --------------------------- ###
def split_h5(filename, cut_points, key=None, save_to_csv=False, frames_to_df=False):
    # df = pd.read_hdf(filename) DOESNT WORK WITH RESULTS_00.h5
    with h5py.File(filename, 'r') as f:

        # GET NUMBER OF FRAMES
        frame_num = f['frames'].shape[0]
        print("Frames in session:", frame_num)

        
        if key == None:
            keys = f.keys()
        else:
            if key not in f.keys():
                print("NOT A VALID KEY")
                print("Please use an available key in this dataset:", f.keys())
                print("QUITTING")
                quit()
            else:
                keys = [key]
        print("keys:",keys)
        
        # LOOP THROUGH ALL KEYS
        for key in keys:
            
    
        # RESHAPING FOR 3D ARRAYS
            if key == 'frames':
                print("[SELECTED]: FRAMES")

                # FRAME DATA
                frames = f[key]
                print(len(frames))
                print(frames.shape)

                # SPLIT BY FRAME
                control_frames = frames[:cut_points[0]]
                stim_frames = frames[cut_points[0]:cut_points[1]]
                post_frames = frames[cut_points[1]:]
                
                print(control_frames)
                print(len(control_frames))
                print("--------------------------------------")
                print(stim_frames)
                print(len(stim_frames))
                print("--------------------------------------")
                print(post_frames)
                print(len(post_frames))
                print("--------------------------------------")
               
                
                # RESHAPE 3D FRAME ARRAY AND CONVERT TO DF
                if frames_to_df == True:
                    print(frames.shape)
                    reshaped_frames = np.reshape(frames, (-1, 80*80))
                    print(reshaped_frames.shape)
                    df = pd.DataFrame(np.array(reshaped_frames))
                    print(df)
                    if save_to_csv == True:
                        # chanege to save as h5
                        df.to_csv(key+".csv", index=False)

                
            
            elif key == 'frames_mask':
                print("[SELECTED]: FRAMES_MASK")
                # FRAME DATA
                frames_mask = f[key]
                print(len(frames_mask))
                print(frames_mask.shape)

                # SPLIT BY FRAME
                control_frames_mask = frames_mask[:cut_points[0]]
                stim_frames_mask = frames_mask[cut_points[0]:cut_points[1]]
                post_frames_mask = frames_mask[cut_points[1]:]
                
                print(control_frames_mask)
                print(len(control_frames_mask))
                print("--------------------------------------")
                print(stim_frames_mask)
                print(len(stim_frames_mask))
                print("--------------------------------------")
                print(post_frames_mask)
                print(len(post_frames_mask))
                print("--------------------------------------")
            

            elif key == 'metadata':
                print("[SELECTED]: METADATA")
                acquisition, extraction, uuid = f[key]['acquisition'], f[key]['extraction'], f[key]['uuid']
                
                # FLIP DATA
                flip_data = extraction['flips']
                print(flip_data)

                # FLIP DF
                flip_df = pd.DataFrame(flip_data)
                print(flip_df)

                # SLICE BY FRAME
                control_flip = flip_data[:cut_points[0]]
                stim_flip = flip_data[cut_points[0]:cut_points[1]]
                post_flip = flip_data[cut_points[1]:]

                # PRINT TO VERIFY
                print(control_flip)
                print(len(control_flip))
                print("--------------------------------------")
                print(stim_flip)
                print(len(stim_flip))
                print("--------------------------------------")
                print(post_flip)
                print(len(post_flip))
                print("--------------------------------------")



            elif key == 'scalars':
                print("[SELECTED]: SCALARS")
                scalars = f[key]
                for i in scalars:

                    # SCALAR DATA
                    scalar_data = scalars[i]
                    print(scalar_data)
                    print("--------------------------------------")

                    # SCALAR DF
                    scalar_data_df = pd.DataFrame(scalar_data)
                    print(scalar_data_df)

                    # SLICE BY FRAME
                    control_scalar = scalar_data[:cut_points[0]]
                    stim_scalar = scalar_data[cut_points[0]:cut_points[1]]
                    post_scalar = scalar_data[cut_points[1]:]

                    print(control_scalar)
                    print(len(control_scalar))
                    print("--------------------------------------")
                    print(stim_scalar)
                    print(len(stim_scalar))
                    print("--------------------------------------")
                    print(post_scalar)
                    print(len(post_scalar))
                    print("--------------------------------------")

            elif key == 'timestamps':
                print("[SELECTED]: TIMESTAMPS")
                timestamps = f[key]
                print(timestamps)
                timestamp_df = pd.DataFrame(timestamps)
                print(timestamp_df)

                control_timestamps = timestamps[:cut_points[0]]
                stim_timestamps = timestamps[cut_points[0]:cut_points[1]]
                post_timestamps = timestamps[cut_points[1]:]
                print("Split Timestamps")
                print("Control:", control_timestamps[0], "-", control_timestamps[-1], "-> frame_num", len(control_timestamps))
                print("Stim:", stim_timestamps[0], "-", stim_timestamps[-1], "-> frame_num", len(stim_timestamps))
                print("Post:", post_timestamps[0], "-", post_timestamps[-1], "-> frame_num", len(post_timestamps))
                


    # df = pd.DataFrame(np.array(h5py.File(filename)['frames']))
    # print(df)


## SPLIT H5 - Can specify key
# split_h5(filename, [18000, 36000])
