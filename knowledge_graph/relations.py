from collections import defaultdict
from itertools import chain
from typing import Dict, List, Callable, Union

import pandas as pd
from pydantic import BaseModel
from toolz import curry, valmap
from toolz.curried import get

from knowledge_graph.types import NodeName, EntityName, RelationName, NodeNameWithUnixTimestamp


class Relation(BaseModel):
    head_entity: EntityName
    tail_entity: EntityName
    relation_type: RelationName
    relations: Dict[NodeName, List[Union[NodeNameWithUnixTimestamp, NodeName]]]
    symmetric: bool


class TimeAwareRelation(Relation):
    def get_all_timestamps(self, sort=True) -> List[int]:
        result = map(get(0), chain(*self.relations.values()))
        if sort:
            result = sorted(result)
        return list(result)


@curry
def iterative_relation_generator(
        source_col: str,
        target_col: str,
        relation_type: str,
        date_col: str,
        symmetric: bool,
        data: pd.DataFrame) -> Relation:
    relations = defaultdict(set)
    columns = [source_col, target_col] + ([date_col] if date_col is not None else [])
    has_date = date_col is not None
    for _, row in data[columns].iterrows():
        if isinstance(row[target_col], list):
            objects = set(row[target_col])
            if has_date:
                objects = set(map(lambda target: (int(row[date_col]), target), objects))
            relations[row[source_col]] |= objects
            if symmetric:
                for object in objects:
                    relations[object[1] if has_date else object].add(
                        (object[0], row[source_col]) if has_date else row[source_col]
                    )
        else:
            relations[row[source_col]].add(
                (int(row[date_col].timestamp()), row[target_col]) if has_date else row[target_col]
            )
            if symmetric:
                relations[row[target_col]].add(
                    (int(row[date_col].timestamp()), row[source_col]) if has_date else row[source_col]
                )
    cls = TimeAwareRelation if has_date else Relation
    return cls(
        head_entity=source_col,
        tail_entity=target_col,
        relation_type=relation_type,
        relations=valmap(list, relations),
        symmetric=symmetric
    )


def get_amazon_relation_generators() -> List[Callable]:
    return [
        iterative_relation_generator('customer', 'product', 'purchase', 'date', True),
        iterative_relation_generator('product', 'also_view', 'also_viewed', None, True),
        iterative_relation_generator('product', 'also_buy', 'also_bought', None, True),
        iterative_relation_generator('product', 'brand', 'produced_by', None, True),
        iterative_relation_generator('product', 'category', 'belongs_to_category', None, True)
    ]


def get_tafeng_relation_generators() -> List[Callable]:
    return [
        iterative_relation_generator('customer', 'product', 'purchase', 'date', True),
        iterative_relation_generator('product', 'zip_code', 'bought_in', None, True),
        iterative_relation_generator('customer', 'age_group', 'belongs_to_age_group', None, True),
        iterative_relation_generator('product', 'product_subclass', 'belongs_to_subclass', None, True)
    ]


def get_x5retail_relation_generators() -> List[Callable]:
    pass
