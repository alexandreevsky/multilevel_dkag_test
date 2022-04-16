from itertools import repeat
from typing import List, Dict, Union
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_graph.datasets import TransrPosNegBatch, CustomerPosNegBatch
from knowledge_graph.layer_generators import LayerNodeGenerator
from models.TransrAggregator import TransrAggregator
from models.config import Config


class Model(nn.Module):
    def __init__(self,
                 config: Config,
                 layer_generators: List[LayerNodeGenerator],
                 device: Optional[torch.device] = None):
        super().__init__()

        self.transR_aggregator = TransrAggregator(config, layer_generators, device)
        self.config = config
        self.n_timesplits = len(layer_generators)
        self.device = device
        mult = self.config.n_layers if self.config.concat_layers else 1
        self.lstm = nn.LSTM(
            input_size=config.entity_embedding_dim * mult,
            hidden_size=config.entity_embedding_dim * mult,
            num_layers=1,
            batch_first=True
        )

    def forward(self,
                one_or_many_batches: Union[List[int], List[Optional[List[int]]]],
                transR: bool):
        if transR:
            return self.transR_aggregator(one_or_many_batches, concat_layers=False, time_start=0)

        return self.forward_entities(one_or_many_batches, time_start=1)

    def forward_entities(self, entities: List[int], time_start: int = 0) -> torch.Tensor:
        customer_embeddings = self._embed_nodes(entities, concat_layers=True, time_start=time_start)

        # (batch_size, entity_embedding_dim)
        _, (h_n, _) = self.lstm(customer_embeddings)
        return h_n.squeeze(0)

    def _embed_nodes(self,
                     nodes: Union[List[int], List[List[int]]],
                     concat_layers: bool,
                     time_start: int = 0,
                     time_end: int = None) -> torch.Tensor:
        node_embeddings = self.transR_aggregator(nodes, concat_layers, time_start, time_end)
        tensors = []

        if not any(isinstance(x, list) for x in nodes):
            nodes = repeat(nodes, len(node_embeddings))

        for time_idx, iter_nodes in enumerate(nodes):
            # (batch_size, 1, entity_embedding_dim) to concatenate further
            tensors.append(
                torch.cat([node_embeddings[time_idx][node].unsqueeze(0)
                           for node in iter_nodes], dim=0).unsqueeze(1)
            )
        # (batch_size, seq_len, entity_embedding_dim)
        return torch.cat(tensors, dim=1)

    def transR_loss(self, batch: List[TransrPosNegBatch], time_outputs: List[Dict[int, torch.Tensor]]) -> torch.Tensor:
        def extract_tensors(timestep_nodes: TransrPosNegBatch, timestep_output: Dict[int, torch.Tensor], attr: str):
            return torch.cat(tuple(map(
                lambda x: timestep_output[x].unsqueeze(0), getattr(timestep_nodes, attr)
            )))

        def project_entities(entity_embeddings: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
            return torch.bmm(entity_embeddings.unsqueeze(1), relation_matrices).squeeze(1)

        def transR_distance(
                projected_heads: torch.Tensor, relation_embeddings: torch.Tensor, projected_tails: torch.Tensor
        ) -> torch.Tensor:
            return torch.sum(torch.pow(projected_heads + relation_embeddings - projected_tails, 2), dim=1)

        def mean_l2_norm(node_embeddings: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.sum(torch.pow(node_embeddings, 2), dim=1))

        loss = 0.0
        for timestep_nodes, timestep_outputs in zip(batch, time_outputs):
            # n_pos_nodes == n_neg_nodes == n_nodes
            # (n_nodes, entity_embedding_dim)
            head_embeddings = extract_tensors(timestep_nodes, timestep_outputs, 'head')
            # (n_nodes, relation_embedding_dim)
            relation_embeddings = self.transR_aggregator \
                .kgat.relation_embedder(torch.tensor(timestep_nodes.relation, device=self.device))
            # (n_nodes, entity_embedding_dim)
            pos_tail_embeddings = extract_tensors(timestep_nodes, timestep_outputs, 'pos_tail')
            # (n_nodes, entity_embedding_dim)
            neg_tail_embeddings = extract_tensors(timestep_nodes, timestep_outputs, 'neg_tail')

            # (n_nodes, entity_embedding_dim, relation_embedding_dim)
            relation_matrices = self.transR_aggregator \
                .kgat.aggregator.relation_matrices[timestep_nodes.relation]

            # (n_nodes, relation_embedding_dim)
            projected_heads = project_entities(head_embeddings, relation_matrices)
            projected_pos_tails = project_entities(pos_tail_embeddings, relation_matrices)
            projected_neg_tails = project_entities(neg_tail_embeddings, relation_matrices)

            # (n_nodes)
            pos_score = transR_distance(projected_heads, relation_embeddings, projected_pos_tails)
            neg_score = transR_distance(projected_heads, relation_embeddings, projected_neg_tails)

            bpr_loss = torch.mean((-1.0) * F.logsigmoid(neg_score - pos_score))

            mean_l2_loss = mean_l2_norm(projected_heads) + mean_l2_norm(projected_pos_tails) \
                           + mean_l2_norm(projected_neg_tails) + mean_l2_norm(relation_embeddings)

            loss += bpr_loss + self.config.transR_l2_weight * mean_l2_loss

        return loss / len(batch)

    def recommender_task_loss(self, batch: CustomerPosNegBatch) -> torch.Tensor:

        loss = 0.0

        # (batch_size, n_timesplits - 1, n_layers*entity_embedding_dim)
        customer_embeddings = self._embed_nodes(
            batch.customer, concat_layers=True, time_start=0, time_end=-1
        )

        # (batch_size, n_timesplits - 1, n_layers*entity_embedding_dim)
        lstm_outputs, _ = self.lstm(customer_embeddings)

        # (batch_size, n_timesplits-1, n_layers*entity_embedding_dim)
        pos_product_embeddings = self._embed_nodes(
            batch.pos_products.T.numpy().tolist(), concat_layers=True, time_start=0, time_end=-1
        )
        pos_product_lstm_outputs, _ = self.lstm(pos_product_embeddings)

        neg_product_embeddings = self._embed_nodes(
            batch.neg_products.T.numpy().tolist(), concat_layers=True, time_start=0, time_end=-1
        )
        neg_product_lstm_outputs, _ = self.lstm(neg_product_embeddings)

        for time_idx in range(self.n_timesplits - 1):
            # (batch_size, 1, n_layers*entity_embedding_dim)
            timestep_lstm_outputs = lstm_outputs[:, time_idx].unsqueeze(1)

            # (batch_size, n_layers*entity_embedding_dim, 1)
            timestep_pos_products = pos_product_lstm_outputs[:, time_idx].unsqueeze(-1)
            timestep_neg_products = neg_product_lstm_outputs[:, time_idx].unsqueeze(-1)

            # (batch_size)
            pos_scores = torch.bmm(timestep_lstm_outputs, timestep_pos_products).squeeze()
            neg_scores = torch.bmm(timestep_lstm_outputs, timestep_neg_products).squeeze()

            loss += torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))

        return loss / (self.n_timesplits - 1)
