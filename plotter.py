import matplotlib.pyplot as plt

def create_column_graph_with_error_bars(means, stds, labels, colors=None, legend_labels=None):
    """
    Create a column graph with error bars using given means, standard deviations, and labels.

    Parameters:
        - means (list): List of means for each column
        - stds (list): List of standard deviations for each column
        - labels (list): List of labels for each column

    Returns:
        - None
    """
    # Set the figure size and resolution for high-quality output
    plt.figure(figsize=(8, 6), dpi=100)

    print(labels[:2])
    print(labels[2:])

    # Create a bar plot with error bars
    if colors is not None:
        bars = plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
    else:
        bar1 = plt.bar(labels[:2], means[:2], yerr=stds[:2], capsize=5, alpha=0.8, color='black', label='Active stim')
        bar2 = plt.bar(labels[2:], means[2:], yerr=stds[2:], capsize=5, alpha=0.8, color='grey', label='Post stim')

    # Add x and y axis labels
    plt.xlabel('Group', fontsize=14)
    plt.ylabel('Locomotive Syllables Counts', fontsize=14)

    # Add a title
    plt.title('Column Graph with Error Bars', fontsize=16)

    # Customize the appearance of the graph
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    plt.ylim(0, max(means) + 50)

    # if colors is not None and legend_labels is not None:
    #     custom_legend = []
    #     for color in set(colors):
    #         idx = colors.index(color)
    #         custom_legend.append(plt.bar(labels[0], means[idx], color=color, alpha=0.8))
    #         print(idx)
    #     plt.legend(custom_legend, ['black', 'grey'], loc='upper right', fontsize=12)
    plt.legend()

    # Show the plot
    
    plt.show()

# Example usage


means = [1029.71428571, 917.76470588, 942.07142857, 799.05882353]
stds = [12.37599839, 11.36774043, 9.65851019, 10.47940038]
labels = ['rTMS - stim', 'sham - stim', 'rTMS - post', 'sham - post']
colors = ['black', 'grey', 'grey', 'grey']
legend_labels = ['black', 'grey']

create_column_graph_with_error_bars(means, stds, labels, colors=None, legend_labels=None)



v_means = [69.16927011, 60.33549952, 62.95426809, 66.4351615,  59.01672798, 61.87328918]
v_stds = [50.38216109, 38.99998818, 53.34181486, 44.79061732, 41.75898958, 43.67893682]
v_labels = ['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim']

create_column_graph_with_error_bars(v_means, v_stds, v_labels)

def plot_bar():
    pass


# means = [1122.78571429, 942.07142857, 1029.71428571, 1073.47058824, 799.05882353, 917.76470588]
# stds = [13.11764614,  9.65851019, 12.37599839, 12.95620819, 10.47940038, 11.36774043]
# labels = ['rTMS - control',  'rTMS - post',  'rTMS - stim',  'sham - control',  'sham - post',  'sham - stim']

# create_column_graph_with_error_bars(means, stds, labels)
