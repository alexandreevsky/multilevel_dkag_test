from typing import Optional

import torch
import torch.nn as nn

from knowledge_graph.types import NodeToEmbedding, Edges
from models.config import Config


class RelationAwareAggregator(nn.Module):
    relation_matrices: torch.Tensor


class RelationAttentiveAggregator(RelationAwareAggregator):
    def __init__(self, config: Config, device: Optional[torch.device]):
        super().__init__()
        self.device = device
        self.relation_matrices = nn.Parameter(torch.Tensor(
            config.n_relations, config.entity_embedding_dim, config.relation_embedding_dim
        ))
        nn.init.xavier_uniform_(self.relation_matrices, gain=nn.init.calculate_gain('relu'))

    def forward(self,
                head_embedding: torch.FloatTensor,
                relation_embedder: nn.Embedding,
                nb_embeddings: NodeToEmbedding,
                edges: Edges) -> torch.Tensor:

        relations = list(map(lambda x: x.relation_type, edges))

        # (n_edges, entity_embedding_dim)
        tail_embeddings = torch.cat([nb_embeddings[to].unsqueeze(0)
                                     for to in map(lambda x: x.to, edges)], dim=0)

        # (n_edges, relation_embedding_dim)
        relation_embeddings = relation_embedder(torch.tensor(relations, device=self.device))
        # (n_edges, entity_embedding_dim, relation_embedding_dim)
        W_r = self.relation_matrices[relations]

        # (n_edges, 1, entity_embedding_dim)
        head_embedding = head_embedding.expand(len(edges), 1, -1)

        # (n_edges, relation_embedding_dim)
        head_values = torch.bmm(head_embedding, W_r).squeeze(1)
        tail_values = torch.tanh(torch.bmm(tail_embeddings.unsqueeze(1), W_r).squeeze() + relation_embeddings)

        # (n_edges)
        scores = torch.bmm(head_values.unsqueeze(1), tail_values.unsqueeze(-1)).view(-1)

        # (n_edges)
        probabilities = torch.softmax(scores, dim=0)

        return (tail_embeddings * probabilities.unsqueeze(1)).sum(dim=0)
