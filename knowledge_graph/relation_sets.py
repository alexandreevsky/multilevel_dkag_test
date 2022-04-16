from itertools import chain
from typing import Dict
from typing import List

from pydantic import BaseModel

from knowledge_graph.relations import Relation, TimeAwareRelation
from knowledge_graph.types import NodeName


class RelationSet(BaseModel):
    relation2idx: Dict[NodeName, int] = None
    idx2relation: Dict[int, NodeName] = None

    def build_vocab(self) -> None:
        self.relation2idx = {relation: idx for idx, relation in enumerate(self.fields_set())}
        self.idx2relation = {idx: relation for relation, idx in self.relation2idx.items()}

    def fields_set(self) -> set:
        ignore_fields = ['relation2idx', 'idx2relation']
        return set(filter(lambda x: x not in ignore_fields, self.__fields_set__))

    def __len__(self):
        return len(self.fields_set())

    def get_all_timestamps(self, sort=True) -> List[int]:
        fields_with_time = list(filter(
            lambda attr: isinstance(getattr(self, attr), TimeAwareRelation),
            self.__fields_set__
        ))
        result = chain(*map(lambda field: getattr(self, field).get_all_timestamps(sort=False), fields_with_time))
        if sort:
            result = sorted(result)
        return list(result)


class AmazonRelationSet(RelationSet):
    also_bought: Relation
    also_viewed: Relation
    produced_by: Relation
    belongs_to_category: Relation
    purchase: TimeAwareRelation


class TaFengRelationSet(RelationSet):
    purchase: TimeAwareRelation
    # bought_together: Relation
    belongs_to_subclass: Relation
    bought_in: Relation  # place has it's own pin code
    belongs_to_age_group: Relation
