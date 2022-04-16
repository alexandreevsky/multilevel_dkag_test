from typing import List, Optional

import torch
import torch.nn as nn
from toolz import valmap

from knowledge_graph.layer_generators import LayerNodeGenerator
from knowledge_graph.types import AdjacencyLists, NodeToEmbedding, Edges
from models.aggregators import RelationAttentiveAggregator
from models.config import Config


class KGAT(nn.Module):
    def __init__(self, config: Config, device: Optional[torch.device]):
        super().__init__()

        self.config = config
        self.device = device
        self.relation_embedder = nn.Embedding(
            num_embeddings=config.n_relations, embedding_dim=config.relation_embedding_dim,
        )

        self.aggregator = RelationAttentiveAggregator(config, device)

        self.activation = nn.LeakyReLU(inplace=True)

        self.node_layer_updating_matrices = nn.ModuleList(
            [nn.Linear(2 * config.entity_embedding_dim, config.entity_embedding_dim)
             for _ in range(config.n_layers)]
        )

    def forward(self,
                entity_embedder: nn.Embedding,
                batch_of_nodes: List[int],
                layer_generator: LayerNodeGenerator,
                concat_layers: bool = False) -> NodeToEmbedding:
        layers, initial_nodes = layer_generator.get_layer_nodes(self.config.n_layers, batch_of_nodes)
        batch_embeddings = self._generate_embeddings(entity_embedder, layers, initial_nodes, concat_layers)

        return batch_embeddings

    def _generate_embeddings(
            self, entity_embedder: nn.Embedding,
            layers: List[AdjacencyLists],
            initial_nodes: List[int],
            concat_layers: bool
    ) -> NodeToEmbedding:
        if concat_layers:
            node_to_embedding = {node: [] for node in layers[-1].keys()}

        previous_embeddings = {node: entity_embedder(
            torch.tensor(node, device=self.device)
        ) for node in initial_nodes}

        for layer_idx in range(len(layers)):
            previous_embeddings = self._process_layer(
                entity_embedder,
                self.node_layer_updating_matrices[layer_idx],
                previous_embeddings,
                layers[layer_idx]
            )
            if concat_layers:
                for node in node_to_embedding:
                    node_to_embedding[node].append(previous_embeddings[node])

        return valmap(lambda tensors: torch.cat(tensors), node_to_embedding) \
            if concat_layers else previous_embeddings

    def _process_layer(self,
                       entity_embedder: nn.Embedding,
                       layer_matrix: nn.Module,
                       previous_layer_embeddings: NodeToEmbedding,
                       current_node_layer_neighbours: AdjacencyLists) -> NodeToEmbedding:
        new_embeddings = {}

        for current_node in current_node_layer_neighbours:
            # if node does not have any neighbours, then it's embedding is equal to the initial
            if len(current_node_layer_neighbours[current_node]):
                neighbourhood_embedding = self._calculate_neighbourhood_embedding(
                    entity_embedder(torch.tensor(current_node, device=self.device)),
                    previous_layer_embeddings,
                    current_node_layer_neighbours[current_node]
                )
                new_node_embedding = self._calculate_new_node_embedding(
                    layer_matrix,
                    previous_layer_embeddings[current_node],
                    neighbourhood_embedding
                )
                new_embeddings[current_node] = new_node_embedding / torch.linalg.norm(new_node_embedding)
            else:
                new_embeddings[current_node] = entity_embedder(torch.tensor(current_node, device=self.device))

        return new_embeddings

    def _calculate_neighbourhood_embedding(self,
                                           source_node_embedding: torch.FloatTensor,
                                           node_embeddings: NodeToEmbedding,
                                           node_neighbours: Edges) -> torch.FloatTensor:

        return self.aggregator(source_node_embedding, self.relation_embedder, node_embeddings, node_neighbours)

    def _calculate_new_node_embedding(self,
                                      updating_matrix: nn.Linear,
                                      node_embedding: torch.Tensor,
                                      neighbourhood_embedding: torch.Tensor) -> torch.Tensor:
        return self.activation(updating_matrix(torch.cat([node_embedding, neighbourhood_embedding])))
