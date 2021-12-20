import numpy as np
import config
import data_converter
import affinity_propogation


def get_clusters():
    data = None
    with open(config.edges_path, 'r') as f:
        data = f.readlines()

    adj_matrix = data_converter.edges_to_adj_matrix(data, config.vertices_amount)
    clusters = affinity_propogation.AP(adj_matrix, config.iterations)
    np.savetxt(config.clusters_path, clusters)
    return clusters