from itertools import repeat
from typing import Union, List, Dict, Optional

import torch
import torch.nn as nn

from knowledge_graph.layer_generators import LayerNodeGenerator
from models.KGAT import KGAT
from models.config import Config


class TransrAggregator(nn.Module):
    def __init__(self,
                 config: Config,
                 layer_generators: List[LayerNodeGenerator],
                 device: Optional[torch.device] = None):
        super().__init__()

        self.kgat = KGAT(config, device)
        self.layer_generators = layer_generators
        self.time_entity_embeddings = nn.ModuleList([
            nn.Embedding(config.n_entities, config.entity_embedding_dim)
            for _ in range(len(layer_generators))
        ])

    def _process_multiple_timesteps(self,
                                    one_or_many_batches: Union[List[int], List[Optional[List[int]]]],
                                    concat_layers: bool,
                                    time_start: int = 0,
                                    time_end: int = None) -> List[Optional[Dict[int, torch.Tensor]]]:

        gens = self.layer_generators[slice(time_start, time_end)]

        if not len(gens):
            raise Exception("Incorrect time offset value")

        results = []
        if not any(isinstance(b, list) for b in one_or_many_batches):
            one_or_many_batches = repeat(one_or_many_batches, len(gens))
        elif len(one_or_many_batches) != len(gens):
            raise Exception(
                f"Incorrect number of batches, {len(gens)} generators != {len(one_or_many_batches)} batches"
            )

        for idx, (batch_nodes, gen) in enumerate(zip(one_or_many_batches, gens)):
            results.append(
                None if batch_nodes is None
                else self.kgat(self.time_entity_embeddings[time_start + idx], batch_nodes, gen, concat_layers)
            )
        return results

    def _process_last_timestep(self, batch: List[int], concat_layers: bool) -> Dict[int, torch.Tensor]:
        return self.kgat(self.time_entity_embeddings[-1], batch, self.layer_generators[-1], concat_layers)

    def forward(self,
                one_or_many_batches: Union[List[int], List[Optional[List[int]]]],
                concat_layers: bool,
                time_start: int = None,
                time_end: int = None,
                ) -> Union[Dict[int, torch.Tensor], List[Optional[Dict[int, torch.Tensor]]]]:
        if time_start is not None:
            return self._process_multiple_timesteps(one_or_many_batches, concat_layers, time_start, time_end)
        return self._process_last_timestep(one_or_many_batches, concat_layers)
