import math
import random
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import joblib
from joblib import Parallel, delayed

def set_thresholds(G, threshold):
    """Set the same activation threshold for all nodes in the graph."""
    for node in G.nodes():
        G.nodes[node]['T'] = threshold
    return G

def reset_states(G):
    """Reset the activation states and edge utilization for all nodes and edges in the graph."""
    for node in G.nodes():
        G.nodes[node]['Trigger'] = None
        G.nodes[node]['Active'] = False
        G.nodes[node]['Active_Neighbor_count'] = -1
        G.nodes[node]['Activation_time'] = -1
    for edge in G.edges():
        G.edges[edge]['Utilization'] = 'N'
    return G

def  make_spread(G, q, max_steps=math.inf, model = 'GI'):
    """Simulate the spread of contagion on the graph until either 90% spread or the step limit is reached."""
    time_step = 0

    considered_nodes = set()
    while calc_spreading_density(G) < 0.9 and time_step < max_steps:
        time_step += 1
        sd_before = calc_spreading_density(G)

        # Count active neighbors for inactive nodes
        for node in G.nodes():
            if not G.nodes[node]['Active']:
                active_neighbors = sum(
                    1 for neighbor in G.neighbors(node) 
                    if G.nodes[neighbor]['Active'] and G.nodes[neighbor]['Activation_time'] < time_step
                )
                G.nodes[node]['Active_Neighbor_count'] = active_neighbors
        
        # Trigger nodes based on neighbor count exceeding the threshold
        for node in G.nodes():
            if G.nodes[node]['Active_Neighbor_count'] >= G.nodes[node]['T'] and not G.nodes[node]['Active']:
                G.nodes[node]['Active'] = True
                G.nodes[node]['Trigger'] = 'T'
                G.nodes[node]['Activation_time'] = time_step
                
                # Mark edges as utilized if neighbors activated earlier
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['Active'] and G.nodes[neighbor]['Activation_time'] < time_step:
                        G.edges[neighbor, node]['Utilization'] = 'T'

        #these are the models that have noise
        if model == 'FIRST' or model == 'CONSTANT' or model == 'SYMMETRIC_FIRST' or model == 'SYMMETRIC_CONSTANT': 
            # Random activation of neighbors based on probability q
            for node in G.nodes():
                if G.nodes[node]['Active'] == False and G.nodes[node]['Active_Neighbor_count'] >= 1 and node not in considered_nodes:
                    if model == 'FIRST' or model == 'SYMMETRIC_FIRST':
                        considered_nodes.add(node)

                    if random.random() < q:
                        if model == 'SYMMETRIC_FIRST' or model == 'SYMMETRIC_CONSTANT':
                            #increase or decrese the threshold by 1 with 50% probability
                            if random.random() < 0.5: 
                                G.nodes[node]['T'] = G.nodes[node]['T']  + 1
                            else:
                                G.nodes[node]['T'] = G.nodes[node]['T']  - 1
                        else:
                            G.nodes[node]['Active'] = True
                            G.nodes[node]['Trigger'] = 'Q'
                            G.nodes[node]['Activation_time'] = time_step
                            
                            for neighbor in G.neighbors(node):
                                if G.nodes[neighbor]['Active'] and G.nodes[neighbor]['Activation_time'] < time_step:
                                    G.edges[neighbor, node]['Utilization'] = 'Q'

        sd_after = calc_spreading_density(G)

        # Early stopping condition if all single exposure nodes used all noisy activations
        all_considered = True
        for node in G.nodes():
            if G.nodes[node]['Active'] == True:
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['Active'] == False and neighbor not in considered_nodes:
                        all_considered = False

        if model == 'GI' and sd_after == sd_before:
            break
        elif (model == 'FIRST' or model == 'SYMMETRIC_FIRST') and all_considered and sd_after == sd_before:
            break

    return G, time_step, calc_spreading_density(G)

def calc_spreading_density(G):
    """Calculate the proportion of active nodes in the graph."""
    active_count = sum(1 for node in G.nodes() if G.nodes[node]['Active'])
    return active_count / len(G.nodes())

def calc_tie_ranges(G):
    """Calculate the tie range (second shortest path length) for each edge in the graph."""
    G_copy = copy.deepcopy(G)
    for edge in G.edges():
        G_copy.remove_edge(*edge)
        try:
            G.edges[edge]['Tie_range'] = nx.shortest_path_length(G_copy, source=edge[0], target=edge[1])
        except nx.NetworkXNoPath:
            G.edges[edge]['Tie_range'] = None
        G_copy.add_edge(*edge)
    return G

def watts_strogatz_rewired(n, k, eta):
    """Generate a Watts-Strogatz graph and adjust rewiring probability based on desired rewiring count."""
    total_edges = n * k // 2  # Total number of edges in the regular ring lattice
    rewiring_prob = eta / total_edges  # Rewiring probability for η rewirings
    return nx.watts_strogatz_graph(n, k, rewiring_prob)

def do_calc_for_Graph(G, eta, q_values, seeds_per_graph, model=None):
    """Simulate the spread of contagion on the graph for different q values and collect results."""
    spread_time, value_pairs = [], []

    G = calc_tie_ranges(G)
    G = set_thresholds(G, 2)
    for q in q_values:
        for i in range(seeds_per_graph):
            G = reset_states(G)
            G = set_thresholds(G, 2)

            seed = random.choice(list(G.nodes()))  # Randomly select a seed node

            # Manually activate the seed node and its neighbors
            G.nodes[seed]['Active'] = True
            G.nodes[seed]['Activation_time'] = 0

            #randomly pic one neighbor of the seed node
            neib = random.choice(list(G.neighbors(seed)))
            G.nodes[neib]['Active'] = True
            G.nodes[neib]['Activation_time'] = 0

            # Simulate the spread
            G, time_step, spreading_density = make_spread(G.copy(), q, max_steps=G.number_of_nodes(), model = model)

            if spreading_density >= 0.9:
                spread_time.append((time_step, eta, q))

            # Collect tie range and utilization for each edge
            for edge in G.edges():
                value_pairs.append((G.edges[edge]['Tie_range'], G.edges[edge]['Utilization'], spreading_density, eta, q))

    return spread_time, value_pairs

n = 500  # Number of nodes
k = 4   # Number of neighbors
q_values = np.linspace(0, 0.025, 10)  # Range of q values

seeds_per_graph = 10
n_jobs = -1  # Use all CPU cores

graphs = [] #create the graphs
for eta in tqdm(range(int(np.sqrt(n)))):
    for i in range(50):
        new_graph = watts_strogatz_rewired(n, k, eta)
        new_graph.graph['eta'] = eta
        graphs.append(new_graph)
    

print('Deterministic')
# Initialize list for value pairs
value_pairs_deterministic = []
spread_time_deterministic= []
results = Parallel(n_jobs=n_jobs)(delayed(do_calc_for_Graph)(G, G.graph['eta'], q_values, seeds_per_graph, model = 'GI') for G in graphs)
for spread_times, value_pairs in results:
    spread_time_deterministic.extend(spread_times)
    value_pairs_deterministic.extend(value_pairs)


print('Constant Exposure')
# Initialize list for value pairs
value_pairs_constant = []
spread_time_log_constant = []
results = Parallel(n_jobs=n_jobs)(delayed(do_calc_for_Graph)(G, G.graph['eta'], q_values, seeds_per_graph, model = 'CONSTANT') for G in graphs)
for spread_times, value_pairs in results:
    spread_time_log_constant.extend(spread_times)
    value_pairs_constant.extend(value_pairs)


print('First Exposure')
# Initialize list for value pairs
value_pairs_first = []
spread_time_log_first = []
results = Parallel(n_jobs=n_jobs)(delayed(do_calc_for_Graph)(G, G.graph['eta'], q_values, seeds_per_graph, model = 'FIRST') for G in graphs)
for spread_times, value_pairs in results:
    spread_time_log_first.extend(spread_times)
    value_pairs_first.extend(value_pairs)

print('Symmetric Noisy Constant')
# Initialize list for value pairs
value_pairs_symmetric_constant = []
spread_time_log_symmetric_constant = []
results = Parallel(n_jobs=n_jobs)(delayed(do_calc_for_Graph)(G, G.graph['eta'], q_values, seeds_per_graph, model = 'SYMMETRIC_CONSTANT') for G in graphs)
for spread_times, value_pairs in results:
    spread_time_log_symmetric_constant.extend(spread_times)
    value_pairs_symmetric_constant.extend(value_pairs)

print('Symmetric Noisy First')
# Initialize list for value pairs
value_pairs_symmetric_first = []
spread_time_log_symmetric_first = []
results = Parallel(n_jobs=n_jobs)(delayed(do_calc_for_Graph)(G, G.graph['eta'], q_values, seeds_per_graph, model = 'SYMMETRIC_FIRST') for G in graphs)
for spread_times, value_pairs in results:
    spread_time_log_symmetric_first.extend(spread_times)
    value_pairs_symmetric_first.extend(value_pairs)

print('done')

# Create DataFrames from the lists of tuples
df_deterministic = pd.DataFrame(value_pairs_deterministic, columns=['Tie range', 'Utilization', 'Spreading_density', 'eta', 'q'])
df_constant = pd.DataFrame(value_pairs_constant, columns=['Tie range', 'Utilization', 'Spreading_density', 'eta', 'q'])
df_first = pd.DataFrame(value_pairs_first, columns=['Tie range', 'Utilization', 'Spreading_density', 'eta', 'q'])
df_symmetric_constant = pd.DataFrame(value_pairs_symmetric_constant, columns=['Tie range', 'Utilization', 'Spreading_density', 'eta', 'q'])
df_symmetric_first = pd.DataFrame(value_pairs_symmetric_constant, columns=['Tie range', 'Utilization', 'Spreading_density', 'eta', 'q'])

#merge the dataframes
df_deterministic['Type'] = 'GI'
df_constant['Type'] = 'CONSTANT'
df_first['Type'] = 'FIRST'
df_symmetric_constant['Type'] = 'SYMMETRIC_CONSTANT'
df_symmetric_first['Type'] = 'SYMMETRIC_FIRST'
df = pd.concat([df_deterministic, df_constant, df_first, df_symmetric_constant, df_symmetric_first], ignore_index=True)

#drop non utilized edges
df = df[df['Utilization'] != 'N']

#save the df to joblib
joblib.dump(df, 'df.joblib')

#create dataframes from the spread times which is a list of tuples
df_spread_time_deterministic = pd.DataFrame(spread_time_deterministic, columns=['Spread_time_90', 'eta', 'q'])
df_spread_time_constant = pd.DataFrame(spread_time_log_constant, columns=['Spread_time_90', 'eta', 'q'])
df_spread_time_first = pd.DataFrame(spread_time_log_first, columns=['Spread_time_90', 'eta', 'q'])
df_spread_time_symmetric_constant = pd.DataFrame(spread_time_log_symmetric_constant, columns=['Spread_time_90', 'eta', 'q'])
df_spread_time_symmetric_first = pd.DataFrame(spread_time_log_symmetric_first, columns=['Spread_time_90', 'eta', 'q'])

joblib.dump(df_spread_time_deterministic, 'df_spread_time_deterministic.joblib')
joblib.dump(df_spread_time_constant, 'df_spread_time_log_constant.joblib')
joblib.dump(df_spread_time_first, 'df_spread_time_log_first.joblib')
joblib.dump(df_spread_time_symmetric_constant, 'df_spread_time_symmetric_constant.joblib')
joblib.dump(df_spread_time_symmetric_first, 'df_spread_time_symmetric_first.joblib')