import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import LabelEncoder

from analysis import read_df, syllable_sort, sum_groups, get_headers
# # Generate some example data
# data = np.random.normal(size=1000)

# # Calculate occupancy and total counts
# occupancy, bins = np.histogram(data, bins=50)
# total_counts = np.sum(occupancy)

# # Calculate PDF
# pdf = occupancy / total_counts

# # Plot normalized occupancy
# plt.bar(bins[:-1], pdf, width=np.diff(bins), color='blue', alpha=0.5)
# plt.xlabel('Data')
# plt.ylabel('Normalized Occupancy')
# plt.title('Normalized Occupancy Plot')
# # plt.show()


# # Define the PDF
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.exp(-0.5*(X**2 + Y**2)) / (2*np.pi)

# # Plot the PDF as a contour plot
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.contour(X, Y, Z)

# print(X)

# # Add a heatmap of the PDF
# sns.heatmap(Z, cmap="YlGnBu", square=True, ax=ax)

# # Set the title and axis labels
# ax.set_title('Normalized 2D Gaussian')
# ax.set_xlabel('x')
# ax.set_ylabel('y')




# # Define x and y coordinates of the probabilities
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)

# # Create a meshgrid of the x and y coordinates
# X, Y = np.meshgrid(x, y)

# # Define the probability density values
# pdf_data = np.random.rand(len(x), len(y))

# # Normalize the probabilities
# pdf_data = pdf_data / np.sum(pdf_data)

# # Create a heatmap of the probability density
# plt.pcolormesh(X, Y, pdf_data, cmap='viridis')
# plt.colorbar()
# plt.show()


moseq_dir = pathlib.Path.cwd().parent
filename = moseq_dir/'data/rTMS/mean_df2.csv'

# df = syllable_sort(read_df(filename), "usage")


# sum_df = sum_groups(syllable_sort(read_df(filename), "usage"))
# print(len(df))

# for i in df:
#     print('---')
#     print(i)
#     print('---')
#     print("control")
#     print(len(df[i]['sham - stim']))
#     print(df[i]['sham - stim'])
#     print("rTMS")
#     print(len(df[i]['rTMS - stim']))
#     print(df[i]['rTMS - stim'])




def organise(df):
    # print(df[0])
    new_df = df
    # for i in range(len(df)):
    #     if df[i][df[0].index("uuid")][:-1] not in new_df:
    #         print(df[i][df[0].index("uuid")][:-1])
    #         new_df[df[i][df[0].index("uuid")][:-1]] = ''
    #     print(df[i][df[0].index("group")], df[i][df[0].index("uuid")], df[i][df[0].index("uuid")][:-1])
    df[0].extend(['treatment', 'timepoint'])
    for row in df[1:]:
        row.extend(row[0].split(" - "))
   
    # print(new_df)
    # print(df[0].index("uuid"))
organise(read_df(filename))

def read_csv(filename):
    df = pd.read_csv(filename)
    return df
print(read_csv(filename))

def add_time_treat(df):
    for idx, key in enumerate(df[0]):
        if len(key.split(" ")) > 1:
            df[0][idx] = key.replace(" ", "_")
    print(df[0])
    df[0].extend(['treatment', 'timepoint'])
    for row in df[1:]:
        row.extend(row[0].split(" - "))
    df = pd.DataFrame(df[1:], columns=df[0])
    return df

dataframe = add_time_treat(read_df(filename))
# dataframe = syllable_sort(add_time_treat(read_df(filename)), "usage")
print(dataframe)













### SYLLABLE ANALYSIS ###

## MAP SYLLABLE NUM TO LABEL
# SYLLABLE_MAP = get_syllable_map(syllable_info_path, info=['label', 'desc'])

# ## PLOT OVERALL SYLLABLE USAGE
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'usage')))

# ## PLOT OVERALL VELOCITY
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'velocity_2d_mm_mean')), titles=['Average Syllable 2d Velocity', 'Syllable', 'Velocity'])
# plot_syllable_sums(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'centroid_x_px_mean')), titles=['Spots', 'Syllable', 'Trace'], plot_type='scatter')

# SYLLABLE LOCATIONS
# arrays, headers = get_location_arrays(sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'centroid_x_px_mean')), sum_groups(syllable_sort(read_df(mean_df_path_usage_nums), 'centroid_y_px_mean')), return_headers=True)
# plot_syllable_locations(arrays, headers)


# # ## PRINT SYLLABLE OVERVIEW
# syllable_dict = syllable_overview(read_df(mean_df_path_usage_nums), ["usage", "duration", "velocity_2d_mm_mean", "velocity_3d_mm_mean", "height_ave_mm_mean", "dist_to_center_px_mean"])
# print_syllable_overview(syllable_dict, names=True)

# ## PLOT SYLLABLE OVERVIEW
# rapid_plot(mean_df_path_usage_prob, 'usage', title='label')
# # rapid_plot(mean_df_path_usage_prob, 'velocity_2d_mm_mean') # velocity goes down
# # rapid_plot(mean_df_path_usage_prob, 'height_ave_mm_mean') # ephrin usually higher than WT

# ## PLOT BEHAVIOUR CLASSIFICATION
# plot_syllable_count(syllable_count(get_syllable_map(syllable_info_path)))

# ## COVARIENCE AND CORRELATIONAL MATRIX
# transition_matrix(transition_matrix_dir, mtype='corr')
# transition_matrix(transition_matrix_dir, mtype='cov')






# def manova(df):

#     # Define dependent variables
#     dependent_vars = ['Var1', 'Var2', 'Var3']

#     # Define independent variable
#     independent_var = 'group'

#     # Encode groups as numeric values
#     encoder = LabelEncoder()
#     groups = encoder.fit_transform(df[independent_var])

#     # Set up data
#     y = df[dependent_vars].to_numpy()

#     # Perform MANOVA
#     manova = MANOVA(endog=y, exog=sm.add_constant(groups))

#     # Print results
#     print(manova.mv_test())

def manova2(df):
    # Load data into a pandas DataFrame
    data = df

    # Separate the independent variables (predictors) and dependent variables (response variables)
    X = data[['usage', 'duration', 'velocity_2d_mm_mean']]
    y = data['syllable']

    # Add intercept column to X
    X = sm.add_constant(X)

    # Fit the MANOVA model
    manova = MANOVA.from_formula('usage + duration + velocity_2d_mm_mean ~ C(group)', data=data)
    result = manova.mv_test()

    # Print the MANOVA results
    print(result.summary())

# manova2(add_time_treat(read_csv(filename)))

def manova_analysis(data, independent_vars, dependent_vars=['usage, velocity_2d_mm_mean']):
    if dependent_vars == "all":
        formula = ' + '.join(data.columns[2:]) + f" ~ {'*'.join(independent_vars)}"
    else:
        formula = ' + '.join(dependent_vars) + f" ~ {'*'.join(independent_vars)}"
        print(formula)
    print("[RUNNING]: MANOVA")
    manova = MANOVA.from_formula(formula, data=data)
    print("[FINISHED]: MANOVA")
    result = manova.mv_test()
    print(result)
    print(result.summary())
    return manova.mv_test()

manova_analysis(add_time_treat(read_df(filename)), ['timepoint', 'treatment'])