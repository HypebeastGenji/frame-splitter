from importlib.resources import path
import os
import pathlib
from statistics import mean
import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pathlib

from utils import get_mouse_id, multi_bar_plot


moseq_dir = pathlib.Path.cwd().parent
data_dir = moseq_dir/'data'

GROUPS = [
        '10Hz WT/',
        '10Hz ephrin/',
        '10Hz WT sham/',
        '10Hz ephrin sham/'
]

def get_group_paths(groups, data_dir):
    return [f"{data_dir}/{group}" for group in groups]

GROUP_PATHS = get_group_paths(GROUPS, data_dir)



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
            session_title = condition_session_title[:condition_session_title.index('(')-1].strip()
            session_condition = condition_session_title[condition_session_title.index('(')+1:-1].strip()
        
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
                    # session_dicts[session_title] = scalar_df
                    session_dicts[session_title][session_condition] = scalar_df

                else:
                    session_dicts[session_title][session.split("-")[0]] = scalar_df
            
            # timestamps = f['timestamps']
            # print(timestamps)
        # break
    print("[EXTRACTED", len(session_dicts), "SESSIONS WITH CONTROL, STIM AND POST]")
    return session_dicts

# ./finals -- The grouped results (control, stim, post)
# ../../ -- raw results


# extracted_dicts = extract_scalars('../data/10Hz WT/', save_to_csv=False, raw=True)
# print(extracted_dicts)



def simple_scalar_analysis(session_dict, group, scalar):
    for session in session_dict:
        # print(session)
        summary = session_dict[session][group].describe()
        # summary = session_dict[session].describe()
        # print(summary)

        # print(session)
        print(summary[scalar])
        
# simple_scalar_analysis(extracted_dicts, 'stim', 'velocity_2d_mm')

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
            # plt.plot(group_condtions, mean_of_means, marker='o')
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
        title = stat.capitalize() + " " + scalar+" per frame"
        multi_bar_plot(stat_list, session_labels, scalar, title)

    return stat_list

# stat_list = scalar_analysis(extracted_dicts, 'velocity_2d_mm', 'mean', session_stats=True, overall_stats=True)


def oneway_anova(data):
    # convert to dict
    groups_dict = {}
    groups = ['control', 'stim', 'post']
    for idx, means in enumerate(data):
        if groups[idx] not in groups_dict:
            groups_dict[groups[idx]] = means
    f_value, p_value = stats.f_oneway(groups_dict['control'], groups_dict['stim'], groups_dict['post'])
    return f_value, p_value

# oneway_anova(stat_list)

def compare_means(groups, scalar, stat='mean'):
    comparison_dict = {}
    for group in groups:
        group_name = group.split('/')[-2]

        if group_name not in comparison_dict:
            comparison_dict[group_name] = {}
        
        extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        # print(extracted_dicts)
        stat_list = scalar_analysis(extracted_dicts, scalar, stat)
        mean_stat = [np.mean(stats_list) for stats_list in stat_list]

        comparison_dict[group_name]['control'] = mean_stat[0]
        comparison_dict[group_name]['stim'] = mean_stat[1]
        comparison_dict[group_name]['post'] = mean_stat[2]

        comparison_dict[group_name]['count'] = len(stat_list[0])

        f_value, p_value = oneway_anova(stat_list)
        comparison_dict[group_name]['p_value'] = p_value

    comparison_df = pd.DataFrame(comparison_dict)
    print(comparison_dict)
    return comparison_df


# comparison_df = compare_means(GROUP_PATHS, 'velocity_2d_mm')
# print(comparison_df)


def get_subject_dict(groups):
    comparison_dict = {}
    for group in groups:
        mouse_dict = {}

        extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        for sessions in extracted_dicts:
            mouse_id = get_mouse_id(sessions)
            if mouse_id not in mouse_dict:
                mouse_dict[mouse_id] = []

        for mouse_id in mouse_dict:
            for session in extracted_dicts:
                if mouse_id in session:
                    # mouse_dict[mouse_id][session] = session
                    # mouse_dict[mouse_id][session] = extracted_dicts[sessions]
                    # print(extracted_dicts[sessions]['control'].describe())
                    # mouse_dict[mouse_id].append(session)
                    mouse_dict[mouse_id].append(extracted_dicts[session])

        group_name = group.split('/')[-2]
        # print(group_name)
        if group_name not in comparison_dict:
            comparison_dict[group_name] = mouse_dict

    return comparison_dict


def compare_subjects(comparison_dict, scalar, stat, plot=False):
    group_stat_dict = {}
    for group in comparison_dict:
        stat_dict = {}
        for mouse in comparison_dict[group]:
            if mouse not in stat_dict:
                stat_dict[mouse] = []
            controls, stims, posts = [], [], []
            mouse_groups = comparison_dict[group][mouse]
            # print(mouse)
            # print(mouse_groups)
            for single_mouse in mouse_groups:
                # print(single_mouse) 
                control_summary = single_mouse['control'].describe()
                stim_summary = single_mouse['stim'].describe()
                post_summary = single_mouse['post'].describe()

                controls.append(control_summary[scalar][stat])
                stims.append(stim_summary[scalar][stat])
                posts.append(post_summary[scalar][stat])
            stat_dict[mouse].append(controls)
            stat_dict[mouse].append(stims)
            stat_dict[mouse].append(posts)
        group_stat_dict[group] = stat_dict
        
        if plot:
            for subject in stat_dict:
                session_labels = []
                stat_list = stat_dict[subject]
                for i in range(len(stat_list[0])):
                    session_labels.append("S" + str(i+1) + " (" + subject + ")")

                title = stat.capitalize() + " " + scalar+" per frame (" + group + ")"
    
                multi_bar_plot(stat_list, session_labels, scalar, title)

    return group_stat_dict


# compare_subjects(get_subject_dict(GROUP_PATHS), 'velocity_2d_mm', 'mean')



mean_df_path = '../data/WT vs EPH (10Hz)/mean_df.csv'
scalar_df_path = '../data/WT vs EPH (10Hz)/scalar_df.csv'

def read_df(filename):
    with open(filename, 'r') as infile:
        data_list = []
        data = infile.readlines()
        for line in data:
            line_data = line.strip().split(',')
            line_data = ["nan" if x == '' else x for x in line_data]
            data_list.append(line_data)
    return data_list

def get_headers(data, quiet=False):
    if not quiet:
        for i in data[0]:
            print(i)
    return data[0]

def read_column(data, column):
    column_idx = data[0].index(column)
    for row in data:
        print(row[column_idx])

def setup_group_dict(data, group_col=0):
    group_dict = {}
    for row in data[1:]:
        if row[group_col] not in group_dict:
            group_dict[row[group_col]] = {}
    print(group_dict)
    return group_dict



def syllable_sort(data, scalar):
    syllable_dict = {}
    for row in data[1:]:
        if row[2] not in syllable_dict:
            syllable_dict[row[2]] = {}
        if row[0] not in syllable_dict[row[2]]:
            syllable_dict[row[2]][row[0]] = [float(row[data[0].index(scalar)])]        
        else:
            syllable_dict[row[2]][row[0]].append(float(row[data[0].index(scalar)]))
    return syllable_dict


def sum_groups(sorted_dict):
    for syllable in sorted_dict:
        for group in sorted_dict.get(syllable):
            sorted_dict[syllable][group] = np.nansum(sorted_dict[syllable][group]) / len(sorted_dict[syllable][group])
    return sorted_dict


def plot_syllable_sums(sorted_dict, titles=['Average Syllable Usage', 'Syllable', 'Usage']):

    means_list = []
    groups = list(sorted_dict['0'].keys())
    for group in groups:
        mean_array = np.zeros(len(sorted_dict))
        means_list.append(mean_array)
    means_array = np.array(means_list)
    
    syllables = []
    for idx1, syllable in enumerate(sorted_dict):
        syllables.append(idx1)
        for idx2, group in enumerate(sorted_dict[syllable]):
            means_array[idx2, idx1] = sorted_dict[syllable][group]

    for idx, group_array in enumerate(means_array):
        plt.plot(syllables, group_array, label=groups[idx], marker='o')
        plt.legend()
    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.show()
    return means_array

# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path), 'usage')))


def syllable_overview(data, scalars):
    overview_dict = {}
    for scalar in scalars:
        sorted_dict = syllable_sort(data, scalar)
        summed_dict = sum_groups(sorted_dict)
        for syllable in summed_dict:
            if syllable not in overview_dict:
                overview_dict[syllable] = {}
            for group in summed_dict[syllable]:
                if group not in overview_dict[syllable]:
                    overview_dict[syllable][group] = {}
                if scalar not in overview_dict[syllable][group]:
                    overview_dict[syllable][group][scalar] = summed_dict[syllable][group]

    return overview_dict


def print_syllable_overview(syllable_dict, syllable_num='all'):
    keys = list(syllable_dict.keys())

    if syllable_num == 'all':
        for syl in syllable_dict:
            syl_df = pd.DataFrame(syllable_dict[syl])
            print("---------------------------------------------------------------------------------------------------")
            print(f"[SYLLABLE: {syl}]")
            print(syl_df)
    elif syllable_num in keys:
        syl_df = pd.DataFrame(syllable_dict[syllable_num])
        print("---------------------------------------------------------------------------------------------------")
        print(f"[SYLLABLE: {syllable_num}]")
        print(syl_df)
    else:
        raise KeyError("Syllable does not exist")
    print("---------------------------------------------------------------------------------------------------")

# syllable_dict = syllable_overview(read_df(scalar_df_path), ["velocity_2d_mm", "velocity_3d_mm", "height_ave_mm", "dist_to_center_px"])

syllable_dict = syllable_overview(read_df(mean_df_path), ["usage", "duration", "velocity_2d_mm_mean", "velocity_3d_mm_mean", "height_ave_mm_mean", "dist_to_center_px_mean"])


print_syllable_overview(syllable_dict)
