import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable

# Use pandas to read the data from the csv file.
df = pd.read_csv('pos.csv')

# Create a dictionary with the data from the csv file.
data = {'x': df['x'], 'y': df['y']}

# Create a dataframe from the dictionary.
df = pd.DataFrame(data)

# Create a list of the x and y coordinates.
x = df['x'].tolist()
y = df['y'].tolist()

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

n = numOfNodes = 20

G = nx.path_graph(0)

for (i, j, a) in zip(x, y, letters):
    for (k, l, b) in zip(x, y, letters):
        if i != k:
            G.add_edge(a, b, weight=math.sqrt((i - k) ** 2 + (j - l) ** 2))

pos = nx.circular_layout(G)
# create a dictionary with the x and y coordinates of the nodes
# Create a dictionary with number than key and zipped with the x and y coordinates.
nodes = {letters[i]: (x[i], y[i]) for i in range(len(x))}

# Draw the graph with the weights as the edge labels and rainbow colors.
nx.draw(G, nodes, edge_color=[G[u][v]['weight'] for (u, v) in G.edges()],
        edge_cmap=plt.cm.rainbow, with_labels=True)
# plt.savefig("prueba.png", transparent=True, dpi=150)
plt.show()

# Copy the graph and add the nodes to it.
G2 = G.copy()

# Delete the edges with weight greater than 250.
G.remove_edges_from([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 200])
# G.add_edge('A', 'B', weight=100)
# Plot the colorbar for the weights.

nx.draw(G, nodes, edge_color=[G[u][v]['weight'] for (u, v) in G.edges()],
        edge_cmap=plt.cm.rainbow, with_labels=True)
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
pesosw = list([G[u][v]['weight'] for (u, v) in G.edges()])
norm = mpl.colors.Normalize(vmin=min(pesosw), vmax=max(pesosw))
cb1 = mpl.colorbar.ColorbarBase(ax=ax_cb,cmap=plt.cm.rainbow,norm=norm,orientation='vertical')
plt.gcf().add_axes(ax_cb)
plt.savefig("reducido.png", transparent=True, dpi=150)
plt.show()

print(ax_n)



# Print the degree of each node.
print(len(G.edges))

# Create a new graph of the minimum spanning tree of the original graph.
MST = nx.minimum_spanning_tree(G)
# Plot the graph.
nx.draw(MST, nodes, with_labels=True)
plt.show()

