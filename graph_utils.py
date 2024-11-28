import dgl
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def create_knn_edges(coordinates, k=5):
    """
    Create edges using K-Nearest Neighbors (KNN).

    Parameters:
    - coordinates (ndarray): 3D coordinates of nodes.
    - k (int): Number of nearest neighbors.

    Returns:
    - src, dst (lists): Source and destination node indices for KNN edges.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    src, dst = [], []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                src.append(i)
                dst.append(neighbor)
    return src, dst


def create_r_sphere_edges(coordinates, r_sphere=1.0):
    """
    Create edges between nodes within a specified spherical distance.

    Parameters:
    - coordinates (ndarray): 3D coordinates of nodes.
    - r_sphere (float): Radius of the sphere to connect nodes.

    Returns:
    - src, dst (lists): Source and destination node indices for R_SPHERE edges.
    """
    num_nodes = coordinates.shape[0]
    src, dst = [], []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if distance <= r_sphere:
                src.append(i)
                dst.append(j)
                src.append(j)
                dst.append(i)
    return src, dst

def create_sequencial_edges(num_nodes):
    src, dst = [], []
    for i in range(num_nodes - 1):
        src.append(i)
        dst.append(i + 1)
        src.append(i + 1)
        dst.append(i)
    return src, dst


def build_heterograph(coordinates, node_features, k=5, r_sphere=1.0, max_len=2000) -> dgl.DGLGraph:
    """
    Build a DGL heterogeneous graph with KNN and R_SPHERE edges.

    Parameters:
    - coordinates (ndarray): 3D coordinates of nodes.
    - node_features (ndarray): L * feat matrix of node information.
    - k (int): Number of nearest neighbors for KNN.
    - r_sphere (float): Radius for R_SPHERE edges.

    Returns:
    - graph (DGLHeteroGraph): A heterogeneous graph with KNN and R_SPHERE edges.
    """
    assert coordinates.shape[0] == node_features.shape[0]
    coordinates = np.array(coordinates)
    # num_nodes = coordinates.shape[0]
    
    num_nodes = min(coordinates.shape[0], max_len)
    coordinates = coordinates[:num_nodes]
    node_features = node_features[:num_nodes]
    # Create KNN edges
    knn_src, knn_dst = create_knn_edges(coordinates, k)
    
    # Create R_SPHERE edges
    r_sphere_src, r_sphere_dst = create_r_sphere_edges(coordinates, r_sphere)

    # Create sequencial edges
    seq_src, seq_dst = create_sequencial_edges(num_nodes)
    
    if num_nodes < max_len:
        pad_len = max_len - coordinates.shape[0]
        coordinates = np.pad(coordinates, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        node_features = np.pad(node_features, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        num_nodes = max_len

    assert coordinates.shape[0] == num_nodes
    assert node_features.shape[0] == num_nodes
    assert num_nodes == max_len
    
    # Define data for heterograph
    data_dict = {
        ('residue', 'knn_edge', 'residue'): (torch.tensor(knn_src), torch.tensor(knn_dst)),
        ('residue', 'r_sphere_edge', 'residue'): (torch.tensor(r_sphere_src), torch.tensor(r_sphere_dst)),
        ('residue', 'seq_edge', 'residue'): (torch.tensor(seq_src), torch.tensor(seq_dst)),
    }
    
    # Create the heterograph
    graph = dgl.heterograph(data_dict, num_nodes_dict={'residue': max_len})
    
    # Add node features
    # graph.ndata['feature'] = torch.tensor(node_features, dtype=torch.float32)
    graph.ndata['x'] = torch.tensor(node_features, dtype=torch.float32)
    
    return graph


def main():
    # Example 3D coordinates
    coordinates = np.random.rand(10, 3)  # 10 nodes with 3D coordinates
    
    # Example node features
    node_features = np.random.rand(10, 7)  # 1D vector of node information
    
    # Parameters for graph construction
    k = 3  # Number of nearest neighbors
    r_sphere = 0.5  # Radius for spherical connections
    
    # Build heterogeneous graph
    hetro_graph = build_heterograph(coordinates, node_features, k, r_sphere)
    
    # Print graph information
    print(hetro_graph)
    print(hetro_graph.ndata)


if __name__ == "__main__":
    main()
