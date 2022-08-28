import h5py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import scipy.stats as scs

# so np arrays dont truncate
import sys
np.set_printoptions(threshold=sys.maxsize)

# filename = "./proc/results_00.h5"
# filename = "../../session_20190203090326 WT4 10Hz/proc/results_00.h5"



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

####### UNCOMMENT
# split_data = dictionary of split values    
# split_data = split_h5(filename, [18000, 36000], print_summary=False)


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



def create_file(old_file, split_data, location, newfile='newfile', check_files=True):
    new_files = ['control', 'stim', 'post']
    for i in range(len(new_files)):
        print('[CREATING FILE]: ' + new_files[i] + '-' + newfile + '.h5')
        with h5py.File(location+'/'+new_files[i] + '-' + newfile + '.h5', 'w') as new_h5:
            with h5py.File(old_file, 'r') as old_h5:
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

                                    # update uuid metadata (appends 'c', 's' or 'p' to end)
                                    uuid = np.array(new_h5['/metadata/uuid']).item()
                                    str_uuid = uuid.decode("utf-8")
                                    new_uuid = str.encode(str_uuid + new_files[i][0])
                                    print("New uuid:", new_uuid)
                                    del new_h5['/metadata/uuid']
                                    new_h5['/metadata/uuid'] = new_uuid

                                    # update sessionName metadata (appends '(control)', '(stim)' or '(post)' to end)
                                    session_name = np.array(new_h5['/metadata/acquisition/SessionName']).item()
                                    str_session_name = session_name.decode("utf-8")
                                    new_session_name = str.encode(str_session_name+ ' ('+new_files[i]+')')
                                    print("New session name:", new_session_name)
                                    del new_h5['/metadata/acquisition/SessionName']
                                    new_h5['/metadata/acquisition/SessionName'] = new_session_name

                                    # update yaml
                                    update_dict = {}
                                    update_dict['uuid'] = new_uuid.decode("utf-8")
                                    update_dict['metadata'] = {'SessionName': new_session_name.decode("utf-8")}
                                    # update_dict['parameters'] = {'flip_classifier': 'largemicewithfibre.pkl'} MAY NEED TO CHANGE AS THIS IS NOT ORIGINAL ONE USED IN EXTRACTION

                                    corresponding_yaml = old_file.replace('.h5', '.yaml')
                                    update_yaml(corresponding_yaml, location, update_dict, newfile=new_files[i] + '-' + newfile)

                print('[NEW FILE CREATED]:', new_files[i]+'-newfile.h5')

    if check_files == True:
        h5print(location+'/'+'control-'+newfile+'.h5')
        print()
        h5print(location+'/'+'stim-'+newfile+'.h5')
        print()
        h5print(location+'/'+'post-'+newfile+'.h5')

####### UNCOMMENT
# create_file(split_data, './finals')

####### UNCOMMENT
# CHECK NEWFILE
# h5print('control-newfile.h5')
# print()
# h5print('stim-newfile.h5')
# print()
# h5print('post-newfile.h5')

def check_finals(filename):
    h5print('./finals/'+filename)
# check_finals('control-results_001.h5')

# update_dict = {uuid: '', metadata: {SessionName: ''}}
def update_yaml(oldfile, destination, update_dict, newfile='newfile'):
    with open(oldfile) as old_yml:
        content = yaml.safe_load(old_yml)

    for key in content:
        if key == 'uuid':
            content[key] = update_dict[key]
        if key == 'metadata':
            if 'SessionName' in update_dict[key]:
                content[key]['SessionName'] = update_dict[key]['SessionName']
        # if key == 'parameters':     ## DONT NEED YET
        #     if 'flip_classifier' in update_dict[key]:
        #         content[key]['flip_classifier'] = update_dict[key]['flip_classifier']

    subdir = destination + '/yamls/'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    print("[CREATING YAML]")
    with open(subdir + newfile + '.yaml', "w") as new_yaml:
        yaml.dump(content, new_yaml, default_flow_style=False)
    print("[YAML CREATED]")

        
# update_dict = {'uuid': 'ligma', 'metadata': {'SessionName': 'balls'}}
# update_yaml('./proc/results_00.yaml', './check', update_dict)


def main(files):
    session_count = 1
    total_sessions = len(files)
    for filename in files:

        new_file_name = filename[:-19].replace(" ", "-")
        file_start = new_file_name.rfind('/') + 1
        new_file_name = new_file_name[file_start:]
        
        print("[STARTING MAIN]: " + new_file_name)
        print("------------------------------------------------------------------------")

        ## READFILE
        print("[READING FILE]: " + str(session_count)+'/'+str(total_sessions))
        h5print(filename)
        print("------------------------------------------------------------------------")

        ## SPLIT FILE
        print("[SPLITTING FILE]: " + str(session_count)+'/'+str(total_sessions))
        split_data = split_h5(filename, [18000, 36000], print_summary=False)
        print("------------------------------------------------------------------------")

        ## CREATE FILES
        # by session count
        # create_file(split_data, './finals', newfile='results_00'+str(session_count))
        # by session name
        
        destination = './finals'
        test_destination = './check' # for testing
        external_destination = '/Volumes/NO NAME/Research/Moseq/Ephrin 10Hz wire'

        create_file(filename, split_data, test_destination, newfile=new_file_name)
        print("------------------------------------------------------------------------")

        session_count += 1
        break # for testing

def start():

    base_dir = '../../' # WT 10Hz wire
    # base_dir = '../../../ephrin 10Hz wire/'  # Ephrin

    files = os.listdir(base_dir)
    session_files = []
    bad_sessions = ['session_20190205095213 WT3 10Hz', 'session_20190206115259 WT6 10Hz'] # WT
    # bad_sessions = ['session_20190218101113 3883 10Hz coil came off', 'session_20190218112308 3907 10Hz coil came off']
    for session in files:
        if session[:7] == 'session' and session not in bad_sessions:
            result_filepath = base_dir + session + '/proc/results_00.h5'
            session_files.append(result_filepath)

    print(session_files)
    confirmation = input("confirm session files? (y/n): ")
    if confirmation.lower() == 'y':
        print('[CONFIRMED]')
        main(session_files)
    elif confirmation.lower() == 'n':
        print('[REJECTED]')
        quit()
    else:
        print("[ERROR]: PLEASE ENTER (y/n): ")
        quit()

start()



def extract_scalars(base_dir, raw=False, save_to_csv=False):
    files = os.listdir(base_dir)
    print("[EXTRACTING SCALARS FROM", len(files), "FILES]")

    if raw == True:
        session_files = []
        bad_sessions = ['session_20190205095213 WT3 10Hz', 'session_20190206115259 WT6 10Hz']
        for session in files:
            if session[:7] == 'session' and session not in bad_sessions:
                result_filepath = base_dir + session + '/proc/results_00.h5'
                session_files.append(result_filepath)
    else:
        session_files = files

    session_dicts = {}
    for session in session_files:
        if raw == True:
            session = session.split(base_dir)[1]
            condition_session_title = session.split('/')[0].replace(' ', '-')
            session_title = condition_session_title
        else:
            condition_session_title = session.split('.')[0]
            session_title = session[session.index("-")+1:-3]

        if session_title not in session_dicts:
            session_dicts[session_title] = {}

        scalar_dict = {}
        with h5py.File(base_dir + '/' + session, 'r') as f:
            scalars = f['scalars']
            for scalar in scalars:
                scalar_data = scalars[scalar]
                scalar_dict[scalar] = scalar_data
            scalar_df = pd.DataFrame(scalar_dict)
            if save_to_csv == True:
                scalar_df.to_csv('./scalar_csv/'+condition_session_title+".csv")
            if session.split("-")[0] not in session_dicts[session_title]:
                if raw == True:
                    session_dicts[session_title] = scalar_df
                else:
                    session_dicts[session_title][session.split("-")[0]] = scalar_df
            
            # timestamps = f['timestamps']
            # print(timestamps)
        # break
    print("[EXTRACTED", len(session_dicts), "SESSIONS WITH CONTROL, STIM AND POST]")
    return session_dicts

# ./finals -- The grouped results (control, stim, post)
# ../../ -- raw results

extracted_dicts = extract_scalars('./finals', save_to_csv=False)

def simple_scalar_analysis(session_dict, group, scalar):
    for session in session_dict:
        summary = session_dict[session][group].describe()
        # print(summary)

        print(session)
        print(summary[scalar])
        
# simple_scalar_analysis(extracted_dicts, 'control', 'velocity_2d_mm')

def scalar_analysis(session_dict, scalar, stat, overall_stats=False, session_stats=False):

    group_condtions = ['control', 'stim', 'post']
    session_titles = []
    stat_list = []
    controls, stims, posts = [], [], []
    
    # GET LIST OF LISTS CONTAINING MEANS
    # should change to be more effiient
    for session in session_dict:
        session_titles.append(session)
        control_summary = session_dict[session]['control'].describe()
        stim_summary = session_dict[session]['stim'].describe()
        post_summary = session_dict[session]['post'].describe()
        # print(session)
        # print(control_summary[scalar]['mean'])
        controls.append(control_summary[scalar][stat])
        stims.append(stim_summary[scalar][stat])
        posts.append(post_summary[scalar][stat])
    stat_list.append(controls)
    stat_list.append(stims)
    stat_list.append(posts)
    

    ## PLOT MEAN OF GROUP MEANS
    if overall_stats == True:
        if stat == 'mean':
            mean_of_means = []
            count = 0
            for group in stat_list:
                print("Mean for", group_condtions[count], "group:", np.mean(group))
                mean_of_means.append(np.mean(group))
                count += 1
            plt.bar(group_condtions, mean_of_means)
            plt.title(stat.capitalize() + " " + scalar+" per frame")
            plt.xlabel("Group Condtions")
            plt.ylabel(scalar)
            # plt.savefig("./WT Analysis/WT "+ scalar +" overall per frame.png")
            plt.show()
        else:
            print("[ERROR]: stat is not mean")

    ## PLOT EACH GROUPS MEANS
    if session_stats == True:
        session_labels = []
        for i in range(len(session_dict)):
            subject_id = session_titles[i].split('-')[1]
            session_labels.append("S" + str(i+1) + " (" + subject_id + ")")
        # session_labels = session_titles # full title (comment out for S1, S2, S3...)
        print(session_labels)

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
        ax.set_title(stat.capitalize() + " " + scalar+" per frame")
        ax.set_xticks(x, session_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()

        # plt.savefig("./WT Analysis/WT "+ scalar +" per frame.png")

        plt.show()

# scalar_analysis(extracted_dicts, 'width_mm', 'mean', session_stats=True, overall_stats=True)



def plot_grouped_scalars(session_dict, group, scalar):
    sessions = []
    for session in session_dict:
        sessions.append(session)
    scalar_list = list(session_dict[sessions[0]][group][scalar])
    print(scalar_list)

    plt.plot(scalar_list)
    plt.show()

# plot_grouped_scalars(extracted_dicts, 'control', 'height_ave_mm')

def raw_extract_scalars():
    return extract_scalars('../../', raw=True, save_to_csv=False)

def plot_normal_dist(domain):
    plt.plot(domain, scs.norm.pdf(domain, 0, 1))
    plt.title("Standard Normal")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

def plot_raw_scalars(session_dict, scalar):
   
    sessions = []
    for session in session_dict:
        sessions.append(session)
    scalar_df = session_dict[sessions[7]][scalar]
    # print(scalar_df)
    scalar_list = list(session_dict[sessions[7]][scalar])
    # print(scalar_list)
    print(max(scalar_list))
    # for dp in scalar_list:
    #     if dp > 60:
    #         # dp = 0
    #         print(dp)
    # z = np.abs(scs.zscore(scalar_list))
    # z = scalar_df.apply(scs.zscore)
    # print(z)
    print(scalar_df)
    scalar_df.to_csv("scalar-velocity.csv")
    scalar_df['z_scores'] = (scalar_df[scalar] - scalar_df[scalar].mean())/scalar_df[scalar].std(ddof=0)
    print(scalar_df)
    scalar_df.to_csv("scalar-velocity-zscores.csv")
    z_scores = scalar_df['z_scores']
    # print(z_scores)
    print(len(z_scores))
    # print(scalar_df.head())
    for col in scalar_df.columns:
        print(col)
    # print(scalar_df[scalar].describe())
    # print(z_scores.describe())


    threshold = 3
    # print(np.where(z_scores > threshold)) 
    outliers_removed = np.where(z_scores < threshold)
    # plt.plot(outliers_removed)


    # plot_normal_dist(z_scores)

    # zplot(np.array(z_scores))


    # plt.plot(scalar_list)
    # plt.vlines(18000, 0, 60, 'grey', '--')
    # plt.vlines(36000, 0, 60, 'grey', '--')
    plt.show()



# plot_raw_scalars(raw_extract_scalars(), 'velocity_2d_mm')


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
