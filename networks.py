import pathlib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

moseq_dir = pathlib.Path.cwd().parent
data_dir = moseq_dir/'data'
transition_matrix_dir = data_dir/'WT vs EPH (10Hz)'/'moseq_output'/'final-model'/'TM bigram'/'WT - control_bigram_transition_matrix.csv'

def plot_transitions(filename):
    df = pd.read_csv(filename, header=None)
    G = nx.from_pandas_adjacency(df)
    nx.draw_circular(G)
    plt.show()

plot_transitions(transition_matrix_dir)
