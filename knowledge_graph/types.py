from typing import Tuple, Dict, List

import torch
from pydantic import BaseModel


class EdgeWithType(BaseModel):
    relation_type: int
    to: int


NodeName = str
EntityName = str
RelationName = str
NodeNameWithUnixTimestamp = Tuple[int, str]
AdjacencyLists = Dict[int, List[EdgeWithType]]
NodeToEmbedding = Dict[int, torch.Tensor]
Edges = List[EdgeWithType]
