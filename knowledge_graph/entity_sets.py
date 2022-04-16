from itertools import chain
from typing import List, Dict

from pydantic import BaseModel


class EntitySet(BaseModel):
    entity2idx: Dict[str, int] = None
    idx2entity: Dict[int, str] = None

    def build_vocab(self) -> None:
        all_values = list(chain(*map(lambda field: getattr(self, field), self.fields_set())))
        self.entity2idx = {entity: idx for idx, entity in enumerate(all_values)}
        self.idx2entity = {idx: entity for entity, idx in self.entity2idx.items()}

    def fields_set(self) -> set:
        ignore_fields = ['entity2idx', 'idx2entity']
        return set(filter(lambda x: x not in ignore_fields, self.__fields_set__))

    def __len__(self):
        return sum(len(getattr(self, field)) for field in self.fields_set())


class TaFengEntitySet(EntitySet):
    customer: List[str]
    age_group: List[str]
    zip_code: List[str]
    product_subclass: List[str]
    product: List[str]


class AmazonEntitySet(EntitySet):
    customer: List[str]
    product: List[str]
    brand: List[str]
    category: List[str]
