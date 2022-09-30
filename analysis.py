import os
from re import M
import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils import get_mouse_id, multi_bar_plot


GROUPS = [
            '../data/10Hz WT/',
            '../data/10Hz ephrin/',
            '../data/10Hz WT sham/',
            '../data/10Hz ephrin sham/'
]

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


extracted_dicts = extract_scalars('../data/10Hz WT/', save_to_csv=False, raw=True)
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
        # print(session_dict)
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
        # print(session_labels)

        title = stat.capitalize() + " " + scalar+" per frame"
    
        multi_bar_plot(stat_list, session_labels, scalar, title)

    return stat_list

stat_list = scalar_analysis(extracted_dicts, 'velocity_2d_mm', 'mean', session_stats=True, overall_stats=True)


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
        mean_stat = [np.mean(stats_list) for stats_list in stat_list]

        comparison_dict[group_name]['control'] = mean_stat[0]
        comparison_dict[group_name]['stim'] = mean_stat[1]
        comparison_dict[group_name]['post'] = mean_stat[2]

        comparison_dict[group_name]['count'] = len(stat_list[0])

        f_value, p_value = oneway_anova(stat_list)
        comparison_dict[group_name]['p_value'] = p_value

    comparison_df = pd.DataFrame(comparison_dict)
    return comparison_df


# comparison_df = compare_means(GROUPS, 'velocity_2d_mm')
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


compare_subjects(get_subject_dict(GROUPS), 'velocity_2d_mm', 'mean')
















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