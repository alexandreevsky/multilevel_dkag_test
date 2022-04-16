from pydantic import BaseModel


class Config(BaseModel):
    entity_embedding_dim: int
    relation_embedding_dim: int
    n_entities: int
    n_relations: int
    n_layers: int
    transR_l2_weight: float
    concat_layers: bool
