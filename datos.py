import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Use pandas to read the data from the csv file.
df = pd.read_csv('nodos.csv')

# Create a dictionary with the data from the csv file.
data = {'key': df['key'], 'x': df['x'], 'y': df['y']}

# Create a dataframe from the dictionary.
df = pd.DataFrame(data)

# Create a list of the x and y coordinates.
x = df['x'].tolist()
y = df['y'].tolist()

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

n = numOfNodes = 20


# create a function to calculate the distance between two points
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


G = nx.path_graph(0)

dio = {i: (j, math.sqrt((a - a2) ** 2 + (b - b2) ** 2)) for i, a, b in zip(data['key'], data['x'], data['y']) for
       j, a2, b2 in zip(data['key'], data['x'], data['y'])}

for (i, j, a) in zip(x, y, letters):
    for (k, l, b) in zip(x, y, letters):
        if i != k:
            G.add_edge(a, b, weight=distance(i, j, k, l))

pos = nx.circular_layout(G)
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
cb1 = mpl.colorbar.ColorbarBase(ax=ax_cb, cmap=plt.cm.rainbow, norm=norm, orientation='vertical')
plt.gcf().add_axes(ax_cb)

plt.savefig("reducido.png", transparent=True, dpi=150)
plt.show()

# Print the degree of each node.
print(len(G.edges))

# Create a new graph of the minimum spanning tree of the original graph.
MST = nx.minimum_spanning_tree(G)
# Plot the graph.
nx.draw(MST, nodes, with_labels=True)
plt.show()


# Write a new dijkstra algorithm that returns the shortest path from a node to another.
def dijkstra(G, start, end):
    # Create a dictionary with the nodes as keys and the distance to the start node as values.
    dist = {v: float('inf') for v in G.nodes()}
    # Set the distance of the start node to 0.
    dist[start] = 0
    # Create a dictionary with the nodes as keys and the previous node as values.
    prev = {v: None for v in G.nodes()}
    # Create a list of the nodes.
    Q = list(G.nodes())
    # While the list of nodes is not empty.
    while Q:
        # Get the node with the minimum distance.
        u = min(Q, key=lambda v: dist[v])
        print(u)
        # Remove the node from the list of nodes.
        Q.remove(u)
        # For each neighbor of the node.
        for v in G[u]:
            # If the distance of the neighbor is greater than the distance of the node plus the weight of the edge.
            if dist[v] > dist[u] + G[u][v]['weight']:
                # Set the distance of the neighbor to the distance of the node plus the weight of the edge.
                dist[v] = dist[u] + G[u][v]['weight']
                # Set the previous node of the neighbor to the node.
                prev[v] = u
    # Return the distance of the end node.
    return dist[end]


# Write a dijkstra algorithm that returns the shortest path from a node all the other nodes.
def dijkstra_all(G, start):
    # Create a dictionary with the nodes as keys and the distance to the start node as values.
    dist = {v: float('inf') for v in G.nodes()}
    # Set the distance of the start node to 0.
    dist[start] = 0
    # Create a dictionary with the nodes as keys and the previous node as values.
    prev = {v: None for v in G.nodes()}
    # Create a list of the nodes.
    Q = list(G.nodes())
    # While the list of nodes is not empty.
    while Q:
        # Get the node with the minimum distance.
        u = min(Q, key=lambda v: dist[v])
        # Remove the node from the list of nodes.
        Q.remove(u)
        # For each neighbor of the node.
        for v in G[u]:
            # If the distance of the neighbor is greater than the distance of the node plus the weight of the edge.
            if dist[v] > dist[u] + G[u][v]['weight']:
                # Set the distance of the neighbor to the distance of the node plus the weight of the edge.
                dist[v] = dist[u] + G[u][v]['weight']
                # Set the previous node of the neighbor to the node.
                prev[v] = u
    # Return the distance of the end node.
    return dist


print(dijkstra_all(G, 'T'))


# Use ant colony optimization to find the shortest path from the start node to the end node.
def ACO(G, start, end):
    # Create a list with the nodes.
    Q = list(G.nodes())
    # Create a dictionary with the nodes as keys and the distance to the start node as values.
    dist = {v: float('inf') for v in G.nodes()}
    # Set the distance of the start node to 0.
    dist[start] = 0
    # Create a dictionary with the nodes as keys and the previous node as values.
    prev = {v: None for v in G.nodes()}
    # Create a list of the nodes.
    Q = list(G.nodes())
    # While the list of nodes is not empty.
    while Q:
        # Get the node with the minimum distance.
        u = min(Q, key=lambda v: dist[v])
        # Remove the node from the list of nodes.
        Q.remove(u)
        # For each neighbor of the node.
        for v in G[u]:
            # If the distance of the neighbor is greater than the distance of the node plus the weight of the edge.
            if dist[v] > dist[u] + G[u][v]['weight']:
                # Set the distance of the neighbor to the distance of the node plus the weight of the edge.
                dist[v] = dist[u] + G[u][v]['weight']
                # Set the previous node of the neighbor to the node.
                prev[v] = u
    # Create a list with the nodes.
    path = [end]
    # While the previous node of the end node is not None.
    while prev[end] is not None:
        # Add the previous node to the list of nodes.
        path.append(prev[end])
        # Set the end node to the previous node.
        end = prev[end]
    # Return the list of nodes.
    return path


print(ACO(G2, 'T', 'A'))
print(nx.dijkstra_path(G2, 'T', 'A'))

# Create a new algorithm that resolve the traveling salesman problem.

# Delete all edges from the nodes that point to itself.


print(len(G2.edges))

a = {i: (j, distance(x1, y1, x2, y2)) for i, x1, y1 in zip(data['key'], data['x'], data['y']) for j, x2, y2 in
     zip(data['key'], data['x'], data['y']) if i != j}
print(a)

# Create a new algorithm that resolve the traveling salesman problem.

#
