"""
This tutorial is from: https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Graphx(object):
    def __init__(self, name="G", n_nodes=6, nname_prefix="node_", edges=None):
        # Initialize the graph
        self.G = nx.Graph(name=name)
        
        self.nname_prefix= nname_prefix
        # Create nodes 
        for i in range(n_nodes):
            self.G.add_node(i, name=nname_prefix + str(i))

        self.n_nodes = n_nodes 

        # Add edges 
        if edges and len(edges) > 0 and len(edges[0])==2:
            self.G.add_edges_from(edges)

    def addEdges(self, edges):
        self.G.add_edges_from(edges)
    
    def addNodes(self, node_ids=None, n_nodes=None):
        """
        Inputs: 
            node_ids (list): node ids to add 
            n_nodes (int): number of new nodes to add 
            node_ids and n_nodes cannot be valid at the same time
        """
        assert (node_ids and n_nodes) == False, "[addNodes] Either n_nodes or node_ids not None!"
        
        if node_ids:
            if isinstance(node_ids, int):    
                node_ids = [node_ids]
        
        if n_nodes:
            node_ids = range(self.n_nodes, self.n_nodes + n_nodes)

        for id in node_ids:
            G.add_node(id, name=self.nname_prefix + str(id))
            self.n_nodes += 1
            

    def __repr__(self):
        return nx.info(self.G)

    
    def showGraph(self):
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show()

if __name__=="__main__":
    n_nodes = 6 
    edges = [(0,1),(0,2),(1,2),(0,3),(3,4),(3,5),(4,5)]

    graph = Graphx(n_nodes=6, edges=edges, nname_prefix="")
    print(f"\n==============\nGraph:\n {graph}")
    print(f"Nodes:\n {graph.G.nodes.data()}")
    
    # Obtain adjacency matrix A 
    A = np.array(nx.attr_matrix(graph.G)[0])
    X = np.array(nx.attr_matrix(graph.G)[1])
    X = np.expand_dims(X, 1)
    print(f"\n==============\nAdj matrix:\n {A}")
    print(f"\n==============\nNode feature matrix:\n {X}")
    
    # Shift operator 
    print(f"A mul X: {A@X}")

    # Take the node feature into account
    graph.G = graph.G.copy() 
    self_loops = [] 
    for id, _ in graph.G.nodes.data():
        self_loops.append((id, id))
    print(self_loops)
    graph.addEdges(edges=self_loops)
    A = np.array(nx.attr_matrix(graph.G)[0])
    print(f"\n==============\nNew Adj matrix:\n {A}")
    
    # Shift operator 
    print(f"A mul X: {A@X}")
     
    # Normalize (D^{-1} A X )
    degs = graph.G.degree()
    D = np.diag([deg for n,deg in degs])
    normalized_AX = np.linalg.inv(D) @ A @ X 
    print(f"D-1 mul A mul X: {normalized_AX}")

    #Symmetrically-normalization
    D_half_norm = fractional_matrix_power(D, -0.5)
    DADX = D_half_norm @ A @ D_half_norm @ X
    print(f"DADX: {DADX}")

    graph.showGraph()
    

