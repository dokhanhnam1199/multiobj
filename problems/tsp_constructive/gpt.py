
import numpy as np
from typing import Set

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: Set[int], distance_matrix: np.ndarray) -> int:
    """
    Select the next node for a constructive TSP tour.
    Return an integer node id in `unvisited_nodes`.
    """
    if not unvisited_nodes:
        return destination_node
    min_distance = float('inf')
    next_node = None
    for node in unvisited_nodes:
        distance = distance_matrix[current_node, node]
        if distance < min_distance:
            min_distance = distance
            next_node = node
    return next_node

