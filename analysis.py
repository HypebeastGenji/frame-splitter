import os
import pathlib
import h5py
import yaml
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pathlib
import pickle

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

from utils import get_line, get_mouse_id, multi_bar_plot, \
                  new_piecewise_linear, piecewise_linear, reorder_list, plot_comparison, avg_df, pickle_save


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



def extract_scalars(base_dir, raw=False, save_to_csv=False, destination='../scalar_csv/'):
    files = os.listdir(base_dir)
    print("[EXTRACTING SCALARS FROM", len(files), "FILES]")

    if raw == True:
        session_files = []
        bad_sessions = ['session_20190205095213 WT3 10Hz', 'session_20190206115259 WT6 10Hz', 'session_20190218101113 3883 10Hz coil came off', 'session_20190218112308 3907 10Hz coil came off', 'session_20190113151744 WT2 Sham', 'session_20190113160334 WT4 Sham', 'session_20190113163631 WT5 Sham', 'session_20190113170857 WT1 Sham', 'session_20190113174353 WT6 Sham', 'session_20190113182616 WT7 Sham', 'session_20190113190625 WT3 Sham', 'session_20190128093115 WT5 sham', 'session_20190128100411 WT6 sham', 'session_20190128103838 WT4 sham', 'session_20190131104916 WT3 Sham coil fell off', 'session_20190131121155 WT5 Sham', 'session_20190131124434 WT6 Sham coil fell off', 'session_20190131162118 WT3 Sham coil fell off', 'session_20190131165326 WT6 Sham coil fell off', 'session_20190117092631 3906 Sham', 'session_20190117100227 3928 Sham', 'session_20190117103940 3876 Sham', 'session_20190117111450 3885 Sham', 'session_20190117114725 3887 Sham', 'session_20190117122252 3897 Sham', 'session_20190128140048 3928 sham', 'session_20190128143448 3876 sham', 'session_20190128150913 3885 sham', 'session_20190128154317 3890 sham', 'session_20190128162613 3895 sham', 'session_20190128173218 3883 sham', 'session_20190128180452 3907 sham', 'session_20190128184018 3918 sham', 'session_20190131142959 3906 Sham', 'session_20190201114235 3890 Sham', 'session_20190201122324 3876 Sham', 'session_20190201130116 should be 3879 Sham', 'session_20190201133253 3883 Sham', 'session_20190201140423 3889 Sham', 'session_20190201144458 3895 Sham']
        for session in files:
            if session[:7] == 'session' and session not in bad_sessions:
                result_filepath = base_dir + session + '/proc/results_00.h5'
                session_files.append(result_filepath)
    else:
        session_files = files

    session_dicts = {}
    count = 0
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
                scalar_df.to_csv(destination + condition_session_title+".csv")
                # print(condition_session_title.split('(')[1][:-1])
            if session.split("-")[0] not in session_dicts[session_title]:
                if raw == True:
                    # session_dicts[session_title] = scalar_df
                    session_dicts[session_title][session_condition] = scalar_df

                else:
                    session_dicts[session_title][session.split("-")[0]] = scalar_df
            
            # timestamps = f['timestamps']
            # print(timestamps)
        # break
        count += 1
    print("[EXTRACTED", len(session_dicts), "SESSIONS WITH CONTROL, STIM AND POST]")
    return session_dicts

# ./finals -- The grouped results (control, stim, post)
# ../../ -- raw results


# extracted_dicts = extract_scalars('../data/10Hz WT/', save_to_csv=False, raw=True)
# print(extracted_dicts)

def save_pickles(groups):
    for group in groups:
        extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        pickle_save(extracted_dicts, data_dir/'scalars'/str(group.split("/")[-2]+'.pickle'))

def get_pickles(group):
    with open(data_dir/'scalars'/str(group[:-1]+'.pickle'), 'rb') as f:
        data = pickle.load(f)
    return data


def plot_extractions(groups, plot_type='plot', extract=False):
    for group in groups:
        if extract:
            extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        else:
            group = group.split('/')[-2]+'/'
            extracted_dicts = get_pickles(group)
        scalar_analysis(extracted_dicts, 'velocity_2d_mm', 'mean', session_stats=True, overall_stats=True, plot=plot_type)


def simple_scalar_analysis(session_dict, group, scalar):
    for session in session_dict:
        # print(session)
        summary = session_dict[session][group].describe()
        # summary = session_dict[session].describe()
        # print(summary)

        # print(session)
        print(summary[scalar])
        
# simple_scalar_analysis(extracted_dicts, 'stim', 'velocity_2d_mm')

def merge_subjects(session_titles, stat_list):
    compare_dict = {}
    group_condtions = ['control', 'stim', 'post']
    session_titles = np.array([get_mouse_id(session_title) for session_title in session_titles])
    stat_array = np.array(stat_list)
    for i in session_titles:
        mask = session_titles == i
        # print(mask)
        titles = session_titles[mask]
        print(titles)
        # data = stat_array[mask]
        # print(data)
        # if len(titles) > 1:
        #     print(titles)
        # print(session_titles[mask])
        # print(mask)
        # print(i)
        # if len(titles)
    # for session_idx, session in enumerate(session_titles):
    #     session_id = get_mouse_id(session)
    #     for condition_idx, data_list in enumerate(stat_list):
    #         if session_id not in compare_dict:
    #             compare_dict[session_id] = {}
    #             if condition_idx not in compare_dict[se]
    #             compare_dict[session_id][group_condtions[condition_idx]] = [data_list[session_idx]]
    #         else:
    #             compare_dict[session_id][group_condtions[condition_idx]].append(data_list[session_idx])
        
    #     #     print(data_list[session_idx])
    #     #     if session_id not in compare_dict:
    #     #         compare_dict[session_id][group_condtions[condition_idx]] = [data_list[session_idx]]
    #     # print("--")
    #         # if session_id not in compare_dict:
    #         #     compare_dict[session_id][group_condtions[condition_idx]] = [data_list[session_idx]]
    #         # else:
    #         #     compare_dict[session_id][group_condtions[condition_idx]].append(data_list[session_idx])
    #         # print(session_id)
    #         # print(session)
    # print(session_titles)
    # print(compare_dict)

    # print(compare_dict)
    # for i in compare_dict:
    #     print(i)
    #     print(len(i))


def scalar_analysis(session_dict, scalar, stat, overall_stats=False, session_stats=False, plot='plot'):

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

        # print(control_summary['velocity_2d_mm'])
        # print("##########")
        # print(session_dict[session]['control'].sem())
        # print(np.mean(session_dict[session]['control']))
        # print(session)
        # print(get_mouse_id(session))

        controls.append(control_summary[scalar][stat])
        stims.append(stim_summary[scalar][stat])
        posts.append(post_summary[scalar][stat])

    # print(len(session_titles))
    # print(len(controls))
    # print(len(posts))
    # print(len(stims))

    stat_list.append(controls)
    stat_list.append(stims)
    stat_list.append(posts)

    # merge_subjects(session_titles, stat_list)


    ## PLOT MEAN OF GROUP MEANS
    if overall_stats == True:
        if stat == 'mean':
            mean_of_means = []
            count = 0
            for group in stat_list:
                # print("Mean for", group_condtions[count], "group:", np.mean(group))
                mean_of_means.append(np.mean(group))
                count += 1
            if plot == 'plot':
                plt.plot(group_condtions, mean_of_means, marker='o')
            else:  
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
        stat_list = scalar_analysis(extracted_dicts, scalar, stat)
        # print(stat_list)
        mean_stat = [np.mean(stats_list) for stats_list in stat_list]

        comparison_dict[group_name]['control'] = mean_stat[0]
        comparison_dict[group_name]['stim'] = mean_stat[1]
        comparison_dict[group_name]['post'] = mean_stat[2]

        comparison_dict[group_name]['count'] = len(stat_list[0])

        f_value, p_value = oneway_anova(stat_list)
        comparison_dict[group_name]['p_value'] = p_value

    comparison_df = pd.DataFrame(comparison_dict)
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
    # print(comparison_dict)
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


def plot_timeseries(groups, scalar):
    for group in groups:
        extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        group_data = []
        group_indices = []
        for session in extracted_dicts:
            # print(extracted_dicts[session][])
            print(session)
            session_data = []
            session_indices = []
            for condition in extracted_dicts[session]:
                # extracted_dicts[session][condition][scalar].plot(label=condition)
                print(condition)
                data = extracted_dicts[session][condition][scalar].to_numpy()
                print(data)
                # session_data.append(data)
                # print(data)
                indices = np.array(extracted_dicts[session][condition][scalar].index.values.tolist())
                print(indices)
            
                plt.plot(indices, data)
                
                
                regression = stats.linregress(indices, y=data)
                regression_line = get_line(indices, regression.slope, regression.intercept)
                plt.plot(indices, regression_line, color='red', linewidth=4)

                stacked = np.stack((indices, data))
                stacked = stacked.T
                print(stacked)
                session_data.append(stacked)
            # print(session_data)
            # print(np.concatenate((session_data[0], session_data[1], session_data[2])))
            session_array = np.concatenate((session_data[0], session_data[1], session_data[2]))
            
                # piecewise_linear(stacked, 10)
                # new_piecewise_linear(stacked, 10)



            plt.legend()
            plt.title(session)
            plt.xlabel('Frames')
            plt.ylabel(scalar.capitalize())
            plt.show()
            break
            plt.plot(extracted_dicts)
        # print(extracted_dicts)

# plot_timeseries(GROUP_PATHS, 'velocity_2d_mm')


def plot_trace(extracted_dict, units='px', plot_style='seperate'): # summary doesnt work yet
    overall_x = {}
    overall_y = {}

    if plot_style == 'summary':
        fig, axes = plt.subplots(len(extracted_dict), 3, sharey=True)
    for row, session in enumerate(extracted_dict):
        if plot_style == 'seperate':
            fig, axes = plt.subplots(1, 3, sharey=True)
        for idx, condition in enumerate(extracted_dict[session]):
            x_vals = extracted_dict[session][condition]['centroid_x_'+str(units)]
            y_vals = extracted_dict[session][condition]['centroid_y_'+str(units)]
            # overall_x.append(x_vals)
            # overall_y.append(y_vals)
            if plot_style == 'seperate':
                axes[idx].plot(x_vals, y_vals)
                axes[idx].set_title(condition)
            elif plot_style == 'summary':
                axes[row, idx].plot(x_vals, y_vals)
                axes[row, idx].set_title(condition)
        if plot_style == 'seperate':
            plt.show()
    if plot_style == 'summary':
        plt.show()

    # # plt.plot(overal_x, overal_y)
    # for i in range(len(overal_x)):
    #     plt.plot(overal_x[i], overal_y[i], label=i)
    # plt.title("Trace")
    # plt.legend()
    # plt.show()



def save_csv(group_paths, destination):

    for group in group_paths:
        if not os.path.exists(destination+'/'+str(group.split('/')[-2])+'/'):
            os.makedirs(destination+'/'+str(group.split('/')[-2])+'/')
        extracted_dicts = extract_scalars(group, save_to_csv=True, raw=True, destination=destination+'/'+str(group.split('/')[-2])+'/')


def converge_csv(group_paths, scalar='velocity_2d_mm', save_csv=False, destination='../scalar_csv/'):
    conditions = ['control', 'stim', 'post']
    converged_list = []

    for group in group_paths:
        converged = {}
        converged_df = { 
            "ID": [],
            "Strain": [],
            "Treatment": [],
            "Timepoint": [],
            "Value": []
        }
        extracted_dicts = extract_scalars(group, save_to_csv=False, raw=True)
        for session in extracted_dicts:
            mouse_id = get_mouse_id(session)
            if mouse_id not in converged:
                converged[mouse_id] = []
                converged[mouse_id].append(extracted_dicts[session])
                # mouse_data.append(extracted_dicts[session])
            else:
                # averaged = converged[mouse_id] 
                converged[mouse_id].append(extracted_dicts[session])
        name = pathlib.PurePath(group).name
        
        for mouse_data in converged:
            if len(converged[mouse_data]) > 1:
                duplicate_data = []
                for duplicates in converged[mouse_data]:
                    condition_data = []
                    for condition in conditions:
                        condition_data.append(duplicates[condition].describe()[scalar]['mean'])
                    duplicate_data.append(condition_data)
                duplicate_array = np.array(duplicate_data)
                mouse_mean = np.mean(duplicate_array, axis=0)
                mean_list = list(mouse_mean)
                converged[mouse_data] = mean_list
                
            else:
                condition_data = []
                for condition in conditions:
                    for single in converged[mouse_data]:
                        condition_data.append(single[condition].describe()[scalar]['mean'])
                converged[mouse_data] = condition_data

        for data in converged:
            for idx, condition in enumerate(conditions):
                converged_df['ID'].append(data)
                if 'WT' in name:
                    converged_df['Strain'].append('WT')
                elif 'ephrin' in name:
                    converged_df['Strain'].append('Ephrin')
                if 'sham' in name:
                    converged_df['Treatment'].append('Sham')
                else:
                    converged_df['Treatment'].append('10Hz')
                converged_df['Timepoint'].append(condition)
                converged_df['Value'].append(converged[data][idx])
        converged_list.append(pd.DataFrame(converged_df))
    final_df = pd.concat(converged_list, axis=0)
    if save_csv:
        final_df.to_csv(destination + 'grouped_scalars_y_px.csv')
    return final_df


# converge_csv(GROUP_PATHS, scalar='centroid_y_px', save_csv=True)


    # print(extracted_dict)
# plot_trace(extract_scalars(GROUP_PATHS[0], save_to_csv=False, raw=True), plot_style='seperate')

## ------------- Syllable Analysis ------------- ##

# mean_df_path = '../data/WT vs EPH (10Hz)/mean_df.csv'
# scalar_df_path = '../data/WT vs EPH (10Hz)/scalar_df.csv'

# mean_df_path_usage_prob = data_dir/'WT vs EPH (10Hz)'/'mean_df.csv'
# mean_df_path_usage_nums = data_dir/'WT vs EPH (10Hz)'/'mean_df_nums.csv'

# mean_df_path_usage_prob = data_dir/'rTMS'/'mean_df2.csv'
mean_df_path_usage_prob = data_dir/'rTMS'/'mean_df_counts.csv'

syllable_info_path = data_dir/'rTMS'/'model'/'syll_info.yaml'


def get_syllable_map(filename, info=['label']):
    with open(filename) as yml:
        content = yaml.safe_load(yml)

    syllable_map = {}
    if len(info) == 1:
        for syl_num in content:
            if syl_num not in syllable_map:
                syllable_map[str(syl_num)] = content[syl_num][info[0]]
    else:
        for idx, item in enumerate(info):
            for syl_num in content:
                if syl_num not in syllable_map:
                    if idx == 0:
                        syllable_map[str(syl_num)] = [content[syl_num][item]]
                    else: 
                        syllable_map[str(syl_num)].append(content[syl_num][item])

    
    return syllable_map

SYLLABLE_MAP = get_syllable_map(syllable_info_path, info=['label', 'desc'])
# print(SYLLABLE_MAP)
# for i in SYLLABLE_MAP.values():
#     print(i)

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

# get_headers(read_df(mean_df_path_usage_prob))

def read_column(data, column):
    column_idx = data[0].index(column)
    for row in data:
        print(row[column_idx])

# read_column(read_df(mean_df_path_usage_prob), "usage")

def setup_group_dict(data, group_col=0):
    group_dict = {}
    for row in data[1:]:
        if row[group_col] not in group_dict:
            group_dict[row[group_col]] = {}
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


def plot_syllable_sums(sorted_dict, plot_type='line', titles=['Average Syllable Usage', 'Syllable', 'Usage']):

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
        if plot_type == 'line':
            plt.plot(syllables, group_array, label=groups[idx], marker='o')
        elif plot_type == 'scatter':
            plt.scatter(syllables, group_array, label=groups[idx], alpha=0.5)
        plt.tick_params(axis='x', labelrotation=90)
        plt.xticks(syllables)
        plt.legend()
    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.show()
    return means_array

# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'usage')))


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


def print_syllable_overview(syllable_dict, syllable_num='all', names=False):
    keys = list(syllable_dict.keys())

    if syllable_num == 'all':
        for syl in syllable_dict:
            syl_df = pd.DataFrame(syllable_dict[syl])
            print("---------------------------------------------------------------------------------------------------")
            if names:
                try:
                    print(f"[SYLLABLE]: {syl}")
                    print(f"[SYLLABLE]: {SYLLABLE_MAP[syl][0]}")
                    print(f"[SYLLABLE]: {SYLLABLE_MAP[syl][1]}")
                except KeyError:
                    break
            else:
                print(f"[SYLLABLE]: {syl}")
            print(syl_df)
    elif syllable_num in keys:
        syl_df = pd.DataFrame(syllable_dict[syllable_num])
        print("---------------------------------------------------------------------------------------------------")
        if names:
            try:
                print(f"[SYLLABLE]: {syllable_num}")
                print(f"[SYLLABLE]: {SYLLABLE_MAP[syllable_num][0]}")
                print(f"[SYLLABLE]: {SYLLABLE_MAP[syl][1]}")
            except KeyError:
                pass
        else:
            print(f"[SYLLABLE: {syllable_num}")
        print(syl_df)
    else:
        raise KeyError("Syllable does not exist")
    print("---------------------------------------------------------------------------------------------------")


# syllable_dict = syllable_overview(read_df(mean_df_path_usage_nums), ["usage", "duration", "velocity_2d_mm_mean", "velocity_3d_mm_mean", "height_ave_mm_mean", "dist_to_center_px_mean"])
# # print(syllable_dict)
# print_syllable_overview(syllable_dict, names=True)

SYLLABLE_SETTINGS = {
        "GO":['walk', 'raise', 'rear', 'inspect', 'drop'],
        "STOP": ['scrunch', 'stop' , 'freeze', 'scratch']
    }


def create_mask(syllable_dict, search_for='GO'):
    keys = list(syllable_dict.keys())
    full_keys = list(SYLLABLE_MAP.keys())
    # print(keys)
    # print(full_keys)

    # print(SYLLABLE_MAP)

    mask = np.logical_not(np.ones(len(keys)))


    for syl in keys:

        if syl in full_keys:
            # print(syl)
            # print(full_keys[int(syl)])
            # print(SYLLABLE_MAP[syl][0])
            if SYLLABLE_MAP[syl][0] in SYLLABLE_SETTINGS[search_for]:
                mask[int(syl)] = True


        # else:
        #     print(syl)
        #     print("-")
        #     print()

    return mask
        
# STAT_DICT {

# }


def add_syllables(syllable_dict, stat, full=False, mask=None, syllable_num='all'):
    syl_stat = []
    if syllable_num == 'all':
        for syl in syllable_dict:
            if full == True:
                if syl in SYLLABLE_MAP:
                    syl_df = pd.DataFrame(syllable_dict[syl])
                    # print(syl_df)
                    syl_stat.append(list(syl_df.iloc[stat]))
            else:
                syl_df = pd.DataFrame(syllable_dict[syl])
                    # print(syl_df)
                syl_stat.append(list(syl_df.iloc[stat]))
    syl_stat = np.array(syl_stat)

    if mask is not None:
        mask_stat = syl_stat[mask]
        return mask_stat
    else:
        return syl_stat
    
def stat_calc(arrays):
    #rTMS - control  rTMS - post  rTMS - stim  sham - control  sham - post  sham - stim
    stat_array = arrays[0]
    counts_array = arrays[1]
    print(counts_array)
    print("---")

    stat_array = stat_array*30

    stat_avg = np.mean(stat_array, axis=0)

    locomotion = stat_array > 30

    locomotion_counts = counts_array * locomotion

    print(locomotion_counts)
    print(locomotion_counts[:, 2])
    print(locomotion_counts[:, 5])

    t_stim_rtms_sham, p_stim_rtms_sham = stats.ttest_rel(locomotion_counts[:, 2], locomotion_counts[:, 5])
    t_post_rtms_sham, p_post_rtms_sham = stats.ttest_rel(locomotion_counts[:, 1], locomotion_counts[:, 4])
    t_post_stim_sham, p_post_stim_sham = stats.ttest_rel(locomotion_counts[:, 5], locomotion_counts[:, 4])
    t_post_stim_rtms, p_post_stim_rtms = stats.ttest_rel(locomotion_counts[:, 2], locomotion_counts[:, 1])

    print(t_stim_rtms_sham, p_stim_rtms_sham)
    # Not sig diff between amount of locomotion syllables in rTMS vs sham during STIM-stimulation (p=0.13427)
    print(t_post_rtms_sham, p_post_rtms_sham)
    # Sig diff between amount of locomotion syllables in rTMS vs sham during POST-stimulation (p=0.0082873)
    print(t_post_stim_sham, p_post_stim_sham)
    # Sig diff between amount of locomotion syllables in stim vs post in SHAM (p=0.0138149)
    print(t_post_stim_rtms, p_post_stim_rtms)
    # Not sig diff between amount of locomotion syllables in stim vs post in rTMS (p=0.0138149)

    locomotion_mean = np.sum(locomotion_counts, axis=0)
    locomotion_std = np.std(locomotion_counts, axis=0)
    print(locomotion_mean)
    print(locomotion_std)



    # print(stat_array)
    anova_array = stat_array
    anova_df = pd.DataFrame(anova_array, columns=['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim'])
    # print(anova_df)
    # print()
    anova_df['Syllable'] = list(anova_df.index)
    df_melted = pd.melt(anova_df, id_vars=['Syllable'], value_vars=['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim'],
                    var_name='Group', value_name='Velocity')
    
    model = mixedlm("Velocity ~ Group", df_melted, groups=df_melted["Syllable"])
    result = model.fit()
    print(result.summary())

    t_stim_rtms_sham2, p_stim_rtms_sham2 = stats.ttest_rel(stat_array[:, 2], stat_array[:, 5])
    t_post_rtms_sham2, p_post_rtms_sham2 = stats.ttest_rel(stat_array[:, 1], stat_array[:, 4])
    t_post_stim_sham2, p_post_stim_sham2 = stats.ttest_rel(stat_array[:, 5], stat_array[:, 4])
    t_post_stim_rtms2, p_post_stim_rtms2 = stats.ttest_rel(stat_array[:, 2], stat_array[:, 1])

    print(t_stim_rtms_sham2, p_stim_rtms_sham2)
    # # Not sig diff between amount of locomotion syllables in rTMS vs sham during STIM-stimulation (p=0.13427)
    print(t_post_rtms_sham2, p_post_rtms_sham2)
    # Sig diff between amount of locomotion syllables in rTMS vs sham during POST-stimulation (p=0.0082873)
    print(t_post_stim_sham2, p_post_stim_sham2)
    # # Sig diff between amount of locomotion syllables in stim vs post in SHAM (p=0.0138149)
    print(t_post_stim_rtms2, p_post_stim_rtms2)
    # # Not sig diff between amount of locomotion syllables in stim vs post in rTMS (p=0.0138149)

    velocity_mean = np.mean(stat_array, axis=0)
    velocity_std = np.std(stat_array, axis=0)
    print(velocity_mean)
    print(velocity_std)

    # f, p = stats.f_oneway(anova_array[:, 0], anova_array[:, 1], anova_array[:, 2], anova_array[:, 3], anova_array[:, 4], anova_array[:, 5])
    # print(f, p)

    # stat_sums = np.sum(stat_array, axis=0)
    
    # f, p = stats.f_oneway(stat_array[:,2], stat_array[:,5])
    # t, pv = stats.ttest_ind(stat_array[:,2], stat_array[:,5])
    # print(f)
    # print(p)
    # print(t)
    # print(pv)
    

    # print(go_sums)
   



    # return syl_stat







def plot_syllable_overview(syllable_dict, scalar, syllable_num='0', rapid_plot=False, title='label'):
    keys = list(syllable_dict.keys())

    if syllable_num in keys:
        groups = []
        scalars = []
        for group in syllable_dict[syllable_num]:
            groups.append(group)
            scalars.append(syllable_dict[syllable_num][group][scalar])

        conditions = [group.split(' - ')[1] for group in groups][:len(groups)//2]
        wild_type = scalars[len(scalars)//2:]
        ephrin = scalars[:len(scalars)//2]

        # reorders lists from control, post, stim to control, stim, post
        conditions, wild_type, ephrin = reorder_list(conditions, wild_type, ephrin, order=[0, 2, 1])

        plt.plot(conditions, wild_type, label="Wild Type", marker='o')
        plt.plot(conditions, ephrin, label="Ephrin", marker='o')
        plt.legend()
        if title == 'label':
            plt.title(f"{scalar.capitalize()} for Syllable {syllable_num}: {SYLLABLE_MAP[syllable_num][0]}")
        elif title == 'desc':
            plt.title(f"{scalar.capitalize()} for Syllable {syllable_num}: {SYLLABLE_MAP[syllable_num][1]}")
        plt.xlabel("Conditions")
        # plt.ylabel("")

        if rapid_plot:
            plt.pause(2)
            plt.clf()
        else:
            plt.show()

# plot_syllable_overview(syllable_dict, "usage", syllable_num='1')

def rapid_plot(filename, scalar, title='label'):
    syllable_dict = syllable_overview(read_df(filename), [scalar])
    for syllable in syllable_dict:
        plot_syllable_overview(syllable_dict, scalar, syllable_num=syllable, rapid_plot=True, title=title)

# rapid_plot(mean_df_path_usage_prob, 'height_ave_mm_mean')


def syllable_count(syllable_map, remove_error=True):
    syllable_dict = {}
    for syllable in syllable_map.values():
        if syllable.strip() not in syllable_dict:
            syllable_dict[syllable.strip()] = 1
        else: 
            syllable_dict[syllable.strip()] += 1
    return syllable_dict


def plot_syllable_count(syllable_count):

    plt.bar(range(len(syllable_count)), list(syllable_count.values()), align='center')
    plt.xticks(range(len(syllable_count)), list(syllable_count.keys()))
    plt.xlabel("Hand-labelled Behavioural Class")
    plt.ylabel("Number of Syllables")
    plt.show()

# plot_syllable_count(syllable_count(get_syllable_map(syllable_info_path)))


transition_matrix_dir = data_dir/'WT vs EPH (10Hz)'/'moseq_output'/'final-model'/'TM bigram'/'WT - control_bigram_transition_matrix.csv'

def transition_matrix(filename, mtype='cov'):
    df = pd.read_csv(filename, header=None)
    # cov
    if mtype == 'cov':
        plt.matshow(df.cov())
    elif mtype == 'corr':
        plt.matshow(df.corr())
    plt.show()

# transition_matrix(transition_matrix_dir)

def get_group_headers(sum_groups):
    return list(sum_groups['0'].keys())


def get_location_arrays(x_vals, y_vals, return_headers=True):
    headers = get_group_headers(x_vals)
    x_list = []
    y_list = []
    for i in x_vals:
        x_row = np.array([x_vals[i][j] for j in x_vals[i]])
        y_row = np.array([y_vals[i][j] for j in x_vals[i]])
        x_list.append(x_row)
        y_list.append(y_row)
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    if return_headers:
        return (x_array, y_array), headers
    else:
        return (x_array, y_array)

def plot_syllable_locations(arrays, headers):
    x_array, y_array = arrays
    for idx, header in enumerate(headers):
        plt.scatter(x_array[:, idx], y_array[:, idx], label=header, alpha=0.5)
    plt.legend()
    plt.xlim(0, 200)
    plt.show()

COLUMNS = ['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim']

def rm_anova(stat_array):
    anova_array = stat_array
    anova_df = pd.DataFrame(anova_array, columns=['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim'])
    # print(anova_df)
    # print()
    anova_df['Syllable'] = list(anova_df.index)
    df_melted = pd.melt(anova_df, id_vars=['Syllable'], value_vars=['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim'],
                    var_name='Group', value_name='Velocity')
    
    model = mixedlm("Velocity ~ Group", df_melted, groups=df_melted["Syllable"])
    result = model.fit()
    print(result.summary())


def t_test(stat_array, t_type='paired'):
    reference_group = stat_array[:, 2]
    cols = [stat_array[:,i] for i in range(len(COLUMNS))]
    if t_type == 'paired':
        t_test_result = []
        for idx, col in enumerate(cols):
            if col is not reference_group:
                t_stat, p_value = stats.ttest_rel(reference_group, col)
                t_test_result.append({'Group': COLUMNS[idx], 'T-statistic': t_stat, 'P-value': p_value})
        for result in t_test_result:
            print(f"Group: {result['Group']}, T-statistic: {result['T-statistic']}, P-value: {result['P-value']}") 
        # t_stim_rtms_sham2, p_stim_rtms_sham2 = stats.ttest_rel(stat_array[:, 2], stat_array[:, 5])



def time_spent(counts, duration):
    print(counts)
    print(duration)
    time_spent = counts * duration
    print(time_spent)
    rm_anova(time_spent)
    



### RAW ANALYSIS ###

# PLOT RAW EXTRACTION DATA
# plot_extractions(GROUP_PATHS, plot_type='plot')
# scalar_analysis()

# ## TABLE WITH MEANS AND ANOVA (ANOVA USE LIST OF MEANS NOT RAW DATA??)
# comparison_df = compare_means(GROUP_PATHS, 'velocity_2d_mm')
# print(comparison_df)

# plot_comparison(comparison_df)

# # ## COMPARE RAW SCALARS WITHIN SUBJECT
# compare_subjects(get_subject_dict(GROUP_PATHS), 'velocity_2d_mm', 'mean', plot=True)

# ## PLOT TIMESERIES OF VELOCITY
# plot_timeseries(GROUP_PATHS, 'velocity_2d_mm')

# ## PLOT TRACE
# plot_trace(extract_scalars(GROUP_PATHS[0], save_to_csv=False, raw=True), plot_style='seperate')


# save_csv(GROUP_PATHS, destination='../scalar_csv')

### SYLLABLE ANALYSIS ###

## MAP SYLLABLE NUM TO LABEL
SYLLABLE_MAP = get_syllable_map(syllable_info_path, info=['label', 'desc'])

# ## PLOT OVERALL SYLLABLE USAGE
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_prob), 'usage')))

# ## PLOT OVERALL VELOCITY
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_prob), 'velocity_2d_mm_mean')), titles=['Average Syllable 2d Velocity', 'Syllable', 'Velocity'])
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'centroid_x_px_mean')), titles=['Spots', 'Syllable', 'Trace'], plot_type='scatter')

# SYLLABLE LOCATIONS
# arrays, headers = get_location_arrays(sum_groups(syllable_sort(read_df(mean_df_path_usage_prob), 'centroid_x_px_mean')), sum_groups(syllable_sort(read_df(mean_df_path_usage_prob), 'centroid_y_px_mean')), return_headers=True)
# plot_syllable_locations(arrays, headers)


# # ## PRINT SYLLABLE OVERVIEW
syllable_dict = syllable_overview(read_df(mean_df_path_usage_prob), ["usage", "duration", "velocity_2d_mm_mean", "velocity_3d_mm_mean", "height_ave_mm_mean", "dist_to_center_px_mean"])
# print_syllable_overview(syllable_dict, names=True)
# go_mask = create_mask(syllable_dict, search_for='GO')
# stop_mask = create_mask(syllable_dict, search_for='STOP')
# stat_calc(add_syllables(syllable_dict, mask=go_mask))
# stat_calc(add_syllables(syllable_dict, mask=stop_mask))
# vel, count
# stat_calc([add_syllables(syllable_dict, 2, full=True), add_syllables(syllable_dict, 0, full=True)])

rm_anova(add_syllables(syllable_dict, 1, full=True))

# counts, duration
time_spent(add_syllables(syllable_dict, 0, full=True), add_syllables(syllable_dict, 1, full=True))

rm_anova(add_syllables(syllable_dict, 5, full=True))
t_test(add_syllables(syllable_dict, 5, full=True))

# ## PLOT SYLLABLE OVERVIEW
# rapid_plot(mean_df_path_usage_prob, 'usage', title='label')
# # rapid_plot(mean_df_path_usage_prob, 'velocity_2d_mm_mean') # velocity goes down
# # rapid_plot(mean_df_path_usage_prob, 'height_ave_mm_mean') # ephrin usually higher than WT

# ## PLOT BEHAVIOUR CLASSIFICATION
# plot_syllable_count(syllable_count(get_syllable_map(syllable_info_path)))

# ## COVARIENCE AND CORRELATIONAL MATRIX
# transition_matrix(transition_matrix_dir, mtype='corr')
# transition_matrix(transition_matrix_dir, mtype='cov')

