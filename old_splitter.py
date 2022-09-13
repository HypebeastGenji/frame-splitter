import h5py
import numpy as np
import pandas as pd

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