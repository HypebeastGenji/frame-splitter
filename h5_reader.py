from time import time
import h5py
import pandas as pd
import numpy as np

# so np arrays dont truncate
import sys
# np.set_printoptions(threshold=sys.maxsize)

filename = "./proc/results_00.h5"

def read_h5():

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[2]
        if isinstance():
            print(a_group_key, "is a group")
        else:
            print(a_group_key, "is not a group")
        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        # print(data)
        

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
    
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        # print(ds_obj)
        # print(ds_arr)

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


def read_with_pandas(filename, cut_points, key=None, save_to_csv=False):
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
            if key == 'frames' or key == 'frames_mask':
                print("[SELECTED]: FRAMES")
                # frames = f[key]
                # print(frames[0])
                # print(frames.shape)
                # reshaped_frames = np.reshape(frames, (-1, 80*80))
                # print(reshaped_frames.shape)
                # print(reshaped_frames.shape)
                # df = pd.DataFrame(np.array(reshaped_frames))
                # print(df)
                # if save_to_csv == True:
                #     # chanege to save as h5
                #     df.to_csv(key+".csv", index=False)
                print("[SKIPPING FRAMES]")
                pass
            

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


## READ WITH H5PY
# h5print(filename)

## READ WITH PANDAS
read_with_pandas(filename, [18000, 36000])


### READ THROUGH h5 FILE AND EVERY SEPERATE AT 2 CUTTING POINTS (frames)
### def cut_frames(filename, cut_points=[18000, 36000])
### ONCE THESE ARRAYS ARE EDITED SAVE AS A NEW H5 file (3 output)