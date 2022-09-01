import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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



for(i,j,a) in zip(x,y,letters):
    for(k,l,b) in zip(x,y,letters):
        if(i != k):
            #print(a + " " + b + " " +str(math.sqrt((i-k)**2 + (j-l)**2)))
            G.add_edge(a, b, weight=math.sqrt((i-k)**2 + (j-l)**2))




pos = nx.circular_layout(G)
#create a diccionary with the x and y coordinates of the nodes
# Create a dictionary with number as key and zipped with the x and y coordinates.
nodes = {letters[i]: (x[i], y[i]) for i in range(len(x))}

# Draw the graph with the weights as the edge labels and rainbow colors.
nx.draw(G, nodes, edge_color=[G[u][v]['weight'] for (u, v) in G.edges()],
        edge_cmap=plt.cm.rainbow, with_labels=True)
#plt.savefig("prueba.png", transparent=True, dpi=150)
plt.show()

# Copy the graph and add the nodes to it.
G2 = G.copy()


# Delete the edges with weight greater than 250.
G.remove_edges_from([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 200])
#G.add_edge('A', 'B', weight=100)

nx.draw(G, nodes, edge_color=[G[u][v]['weight'] for (u, v) in G.edges()],
        edge_cmap=plt.cm.rainbow, with_labels=True)
plt.savefig("reducido.png", transparent=True, dpi=150)
plt.show()


# Print the degree of each node.
print(len(G.edges))


print(len(pred))

# Create a new graph of the minimum spanning tree of the original graph.
MST = nx.minimum_spanning_tree(G)
# Plot the graph.
nx.draw(MST, nodes, with_labels=True)
plt.show()



# Create a new graph using arista as the edges of the new graph.
#G2 = nx.Graph()
#for(i,j,a) in zip(aristas,pesos):
#    G2.add_edge(i, j, weight=a)
#
#nx.draw(G2, nodes, with_labels=True)
#plt.show()
# Delete the edges directed from the node to itself.
# Plot the graph.





# create a new graph with the diskstra algorithm as edges and the nodes of the original graph.
#H = nx.Graph()
#H.add_nodes_from(G.nodes())
#H.add_weighted_edges_from([(u, v, d['weight']) for (u, v, d) in zip(pred.keys(), pred.values(), dist.values())])

