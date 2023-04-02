import networkx as nx
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import plotly.graph_objects as go

moseq_dir = pathlib.Path.cwd().parent

rtms_matrix = moseq_dir/'data/rTMS/model/rTMS - stim_bigram_transition_matrix.csv'
sham_matrix = moseq_dir/'data/rTMS/model/sham - stim_bigram_transition_matrix.csv'
rtms_counts =  moseq_dir/'data/rTMS/model/rTMS - stim_syllable_counts.csv'
sham_counts =  moseq_dir/'data/rTMS/model/sham - stim_syllable_counts.csv'

def plot_transitions(matrix, counts, layout='circular', read=True, change=False):
    # Read in the transition matrix from CSV file
    if read:
        df = pd.read_csv(matrix, header=None)
        counts = pd.read_csv(counts)
    else:
        df = matrix
        counts = counts

    # Convert pandas DataFrame to a NetworkX directed graph object
    G = nx.DiGraph()
    G = nx.from_pandas_adjacency(df, create_using=G)

    # Set up the circular layout
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G)

    # Get the edge weights as a dictionary
    weights = nx.get_edge_attributes(G, 'weight')
    if change:
        edge_colors = ['blue' if w > 0 else 'red' for w in weights.values()]
        weights = {k: v * -1 for k, v in weights.items() if v < 0}


    # Normalize the weights to values between 0 and 1
    min_weight = min(weights.values())
    max_weight = max(weights.values())
    weights_normalized = {k: (v - min_weight) / (max_weight - min_weight) for k, v in weights.items()}

    # Normalize the weights to either 0.1, 1, or 2.5
    # weights_normalized = {k: 0.1 if v < 0.33 else (1 if v < 0.67 else 2.5) for k, v in weights_normalized.items()}

    
    node_sizes = {row['# syllable id']: row['counts'] for _, row in counts.iterrows()}
    edge_list = [(u, v) for u, v in G.edges()]

    # Draw the network graph with edge weights as thickness
    node_sizes = [node_sizes.get(node, 800) for node in G.nodes()]
    if change:
        nx.draw(G, pos, with_labels=True, font_size=8, node_size=node_sizes, arrowsize=10, arrowstyle='-', alpha=1, node_color='w', edgecolors='black', edgelist=edge_list, edge_color=edge_colors, width=list(weights_normalized.values()))
    else:
        nx.draw(G, pos, with_labels=True, font_size=8, node_size=node_sizes, arrowsize=10, arrowstyle='-', alpha=1, node_color='w', edgecolors='black', edgelist=edge_list, edge_color=list('b'), width=list(weights_normalized.values()))

    # Show the plot
    plt.show()

plot_transitions(sham_matrix, sham_counts)
plot_transitions(rtms_matrix, rtms_counts)



def calc_diff(matrices, counts): 
    matrix1, matrix2 = pd.read_csv(matrices[0], header=None), pd.read_csv(matrices[1], header=None)
    count1, count2 = pd.read_csv(counts[0]), pd.read_csv(counts[1])

    prob_change = matrix2 - matrix1
   
    count_change = pd.DataFrame()
    count_change["# syllable id"] = count1["# syllable id"]
    count_change["counts"] = count2["counts"] - count1["counts"]

    return prob_change, count_change

prob_diff, count_diff = calc_diff([rtms_matrix, sham_matrix], [sham_counts, rtms_counts])
print(count_diff)
# need to make counts positive
plot_transitions(prob_diff, count_diff, read=False, change=True, layout="circular")


