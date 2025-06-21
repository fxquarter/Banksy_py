import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Union, Tuple, Optional
import anndata

from banksy_utils.time_utils import timer
from banksy.csr_operations import remove_greater_than, row_normalize

@timer
def construct_spatial_graph(
    locations: np.ndarray,
    k: Optional[int] = None,
    radius: Optional[float] = None,
    nbr_object: Optional[NearestNeighbors] = None
) -> csr_matrix:
    """
    Constructs a spatial graph from locations based on k-neighbors and/or radius.

    This function can operate in three modes:
    1. k-NN mode: If `k` is provided and `radius` is None.
    2. Radius mode: If `radius` is provided and `k` is None.
    3. k-NN with radius pruning: If both `k` and `radius` are provided.

    Args:
        locations: A numpy array of shape (n_samples, n_features) with coordinates.
        k: The number of nearest neighbors to find for each point.
        radius: The radius to find neighbors within. Also used for pruning the k-NN graph.
        nbr_object: An optional pre-fitted NearestNeighbors object.

    Returns:
        A CSR matrix representing the spatial distance graph.
    """
    if k is None and radius is None:
        raise ValueError("Either 'k' or 'radius' must be provided.")

    if nbr_object is None:
        nbr_object = NearestNeighbors(algorithm='ball_tree').fit(locations)

    if k is None:
        # Radius-based graph
        graph = nbr_object.radius_neighbors_graph(radius=radius, mode="distance")
    else:
        # k-NN graph, possibly with radius pruning
        graph = nbr_object.kneighbors_graph(n_neighbors=k, mode="distance")
        if radius is not None:
            # The remove_greater_than function in csr_operations modifies the graph in place
            # if copy=False, so we don't need to reassign.
            remove_greater_than(graph, radius, copy=False, verbose=False)
    
    return graph

@timer
def theta_from_spatial_graph(locations: np.ndarray,
                             spatial_graph: csr_matrix,
                             ) -> csr_matrix:
    """
    get azimuthal angles from spatial graph and coordinates
    (assumed dim 1: x, dim 2: y, dim 3: z...)

    returns CSR matrix with theta (azimuthal angles) as .data
    """

    theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

    for n in range(spatial_graph.indptr.shape[0] - 1):
        ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
        nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

        if nbr_indices.size > 0:
            self_coord = locations[[n], :]
            nbr_coord = locations[nbr_indices, :]
            relative_coord = nbr_coord - self_coord

            theta_data[ptr_start:ptr_end] = np.arctan2(
                relative_coord[:, 1], relative_coord[:, 0])

    theta_graph = spatial_graph.copy()
    theta_graph.data = theta_data

    return theta_graph


@timer
def generate_spatial_weights(
    locations: np.ndarray,
    num_neighbours: int = 10,
    m: int = 0,
    decay_type: str = "scaled_gaussian",
    max_radius: Optional[Union[float, str]] = 'auto',
    radius_multiplier: float = 3.0,
    nbr_object: Optional[NearestNeighbors] = None,
    verbose: bool = True,
) -> Tuple[csr_matrix, csr_matrix, Optional[csr_matrix]]:
    """
    Generate a graph (csr format) where edge weights decay with distance.
    This is an optimized function that combines k-NN with radius pruning.

    Args:
        locations: Coordinates of spots/cells.
        num_neighbours: Number of nearest neighbors to consider.
        m: Azimuthal fourier transform order.
        decay_type: How weights decay with distance.
        max_radius: Maximum connection radius. If 'auto', it is determined
                    from the median distance of neighbors. If None, no radius
                    pruning is performed.
        radius_multiplier: Multiplier for the median distance to set auto radius.
        nbr_object: Pre-fitted NearestNeighbors object.
        verbose: Verbosity level.

    Returns:
        A tuple of (weighted_graph, distance_graph, theta_graph).
    """

    if nbr_object is None:
        if verbose: print("Building BallTree for nearest neighbor search...")
        nbr_object = NearestNeighbors(algorithm='ball_tree').fit(locations)

    effective_k = num_neighbours * (m + 1)
    
    final_radius = None
    if max_radius == 'auto':
        if verbose: print("Determining adaptive radius...")
        # Find distance to the num_neighbours nearest neighbor for auto radius calculation
        # We query for k+1 because the point itself is included at distance 0.
        distances, _ = nbr_object.kneighbors(n_neighbors=num_neighbours + 1)
        # Use the k-th neighbor distance (column at index num_neighbours)
        median_dist = np.median(distances[:, num_neighbours])
        final_radius = median_dist * radius_multiplier
        if verbose: print(f"Adaptive radius set to {final_radius:.3f}")
    elif isinstance(max_radius, (int, float)):
        final_radius = max_radius

    distance_graph = construct_spatial_graph(
        locations,
        k=effective_k,
        radius=final_radius,
        nbr_object=nbr_object
    )

    if m > 0:
        theta_graph = theta_from_spatial_graph(locations, distance_graph)
    else:
        theta_graph = None

    graph_out = distance_graph.copy()

    # compute weights from nbr distances (r)
    if decay_type == "uniform":
        graph_out.data = np.ones_like(graph_out.data)
    elif decay_type == "reciprocal":
        # Add epsilon to avoid division by zero
        graph_out.data = 1 / (graph_out.data + 1e-9)
    elif decay_type == "reciprocal_squared":
        graph_out.data = 1 / (graph_out.data**2 + 1e-9)
    elif decay_type == "scaled_gaussian":
        indptr, data = graph_out.indptr, graph_out.data
        for n in range(len(indptr) - 1):
            start_ptr, end_ptr = indptr[n], indptr[n + 1]
            if end_ptr > start_ptr:
                nbrs = data[start_ptr:end_ptr]
                median_r = np.median(nbrs)
                if median_r > 1e-9:
                    weights = np.exp(-(nbrs / median_r) ** 2)
                    data[start_ptr:end_ptr] = weights
                else:
                    # If median is zero, all distances are zero, so weights are 1.
                    data[start_ptr:end_ptr] = 1.0

    else:
        raise ValueError(
            f"Weights decay type <{decay_type}> not recognised."
        )

    row_normalize(graph_out, verbose=False) # Keep verbose off to avoid spam

    if m > 0 and theta_graph is not None:
        graph_out.data = graph_out.data * np.exp(1j * m * theta_graph.data)

    return graph_out, distance_graph, theta_graph

@timer
def median_dist_to_nearest_neighbour(adata: anndata.AnnData,
                                     key: str = "coord_xy"):
    
    '''Finds and returns median cell distance in a graph'''
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata.obsm[key])
    distances, indices = nbrs.kneighbors(n_neighbors=2) # k=2 to get the first nearest neighbor
    median_cell_distance = np.median(distances[:, 1])
    print(f"\nMedian distance to closest cell = {median_cell_distance:.4f}\n")
    return nbrs 