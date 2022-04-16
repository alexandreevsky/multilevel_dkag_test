from itertools import chain
from typing import List, Tuple

from numpy.random import choice

from knowledge_graph.types import AdjacencyLists


class LayerNodeGenerator:
    def __init__(self, adj_lists: AdjacencyLists, n_neighbours: int):
        self.adj_lists = adj_lists
        self.n_neighbours = n_neighbours

    def generate_neighbours(self, cur_layer_nodes: List[int]) -> AdjacencyLists:
        neigbours = {}
        for node in cur_layer_nodes:
            node_nb = self.adj_lists[node]
            if len(node_nb) >= self.n_neighbours:
                node_nb = choice(node_nb, size=self.n_neighbours, replace=False).tolist()
            neigbours[node] = node_nb
        return neigbours

    def get_layer_nodes(self, n_layers: int, batch_of_nodes: List[int]) -> Tuple[List[AdjacencyLists], List[int]]:

        last_layer_nodes = list(batch_of_nodes)
        layers = []

        for layer_idx in range(n_layers):
            layers.insert(0, self.generate_neighbours(last_layer_nodes))
            last_layer_nodes = list(
                set(layers[0].keys()) | set(map(
                    lambda x: x.to, chain(*layers[0].values())
                ))
            )

        return layers, last_layer_nodes
