import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from itertools import chain
from typing import List, Tuple, Dict, Callable

import networkx as nx
import pandas as pd
from toolz import valmap, valfilter

from knowledge_graph.entity_sets import AmazonEntitySet, TaFengEntitySet, EntitySet
from knowledge_graph.relation_sets import AmazonRelationSet, TaFengRelationSet, RelationSet
from knowledge_graph.relations import TimeAwareRelation, Relation
from knowledge_graph.types import NodeName, AdjacencyLists, EdgeWithType


class KnowledgeGraph(ABC):
    entity_set: EntitySet
    relation_set: RelationSet

    @abstractmethod
    def _load_entities(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _load_relations(self, *args, **kwargs) -> None:
        pass

    def _filter_relations(self,
                          relation_subset: Relation,
                          after_unix: int, before_unix: int) -> Dict[NodeName, List[NodeName]]:
        relations = relation_subset.relations
        if isinstance(relation_subset, TimeAwareRelation):
            relations = valmap(lambda tos: list(map(
                lambda filtered: filtered[1],
                filter(lambda node_with_timestamp: after_unix <= node_with_timestamp[0] < before_unix, tos)
            )), relations)
        return valfilter(len, relations)

    def to_adj_lists(self, after: datetime = None, before: datetime = None) -> AdjacencyLists:
        adj_lists = defaultdict(list)
        entity2idx = self.entity_set.entity2idx

        after_unix = int(datetime(1989, 1, 1).timestamp()) if after is None else int(after.timestamp())
        before_unix = int(datetime(datetime.now().year + 1, 1, 1).timestamp()) \
            if before is None else int(before.timestamp())

        for relation_subset in map(lambda type: getattr(self.relation_set, type), self.relation_set.fields_set()):
            relation_idx = self.relation_set.relation2idx[relation_subset.relation_type]
            self._logger.info(f'converting {relation_subset.relation_type}')
            filtered_relations = self._filter_relations(relation_subset, after_unix, before_unix)
            for node_name in filtered_relations:
                for edge in map(lambda x: EdgeWithType(relation_type=relation_idx, to=entity2idx[x]),
                                filtered_relations[node_name]):
                    adj_lists[entity2idx[node_name]].append(edge)
                    # if relation_subset.symmetric:
                    #     adj_lists[edge.to].append(EdgeWithType(relation_type=relation_idx, to=entity2idx[node_name]))

        return adj_lists

    def to_networkx_graph(self, after: datetime = None, before: datetime = None) -> nx.DiGraph:
        adj_lists = self.to_adj_lists(after, before)
        di_graph = nx.DiGraph()
        nodes = set()
        edge_list = []
        for source, edges in adj_lists.items():
            targets = set(map(lambda x: x.to, edges))
            nodes.add(source)
            nodes |= targets
            edge_list.extend([(source, t) for t in targets])

        di_graph.add_nodes_from(nodes)
        di_graph.add_edges_from(edge_list)

        return di_graph


class AmazonGraph(KnowledgeGraph):
    _logger = logging.getLogger('AmazonGraph')

    def __init__(self,
                 dataframes: Dict[str, pd.DataFrame],
                 relation_generators: List[Tuple[str, Callable]]):
        super().__init__()
        self._load_entities(dataframes['reviews'], dataframes['meta'])
        self._load_relations(dataframes, relation_generators)

    def _flatten_column(self, col: pd.Series) -> set:
        return set(chain(*col.values.tolist()))

    def _product_unique(self, reviews: pd.DataFrame, meta: pd.DataFrame) -> set:
        return set(reviews['product']) | self._flatten_column(meta.also_buy) | self._flatten_column(meta.also_view)

    def _load_entities(self, reviews: pd.DataFrame, meta: pd.DataFrame) -> None:
        self._logger.info('loading entities')
        self.entity_set = AmazonEntitySet(
            customer=reviews.customer.unique().tolist(),
            product=list(self._product_unique(reviews, meta)),
            brand=meta.brand.unique().tolist(),
            category=list(set(chain(*meta.category.values.tolist()))),
        )
        self.entity_set.build_vocab()

    def _load_relations(self,
                        amazon_frames: Dict[str, pd.DataFrame],
                        relation_generators: List[Tuple[str, Callable]]) -> None:
        self._logger.info('loading relations')
        relations = {}
        for frame_name, generator in relation_generators:
            relation = generator(amazon_frames[frame_name])
            self._logger.info(f'loaded {relation.relation_type}')
            relations[relation.relation_type] = relation
        self.relation_set = AmazonRelationSet(**relations)
        self.relation_set.build_vocab()


class TaFengGraph(KnowledgeGraph):
    _logger = logging.getLogger('TaFengGraph')

    def __init__(self, dataframe: pd.DataFrame, relation_generators: List[Callable]):
        super().__init__()
        self._load_entities(dataframe)
        self._load_relations(dataframe, relation_generators)

    def _load_entities(self, dataframe: pd.DataFrame) -> None:
        self._logger.info('loading entities')
        entities = {}
        for col in dataframe.columns:
            entities[col] = dataframe[col].unique().tolist()
        self.entity_set = TaFengEntitySet(**entities)
        self.entity_set.build_vocab()

    def _load_relations(self, dataframe: pd.DataFrame, relation_generators: List[Callable]) -> None:
        self._logger.info('loading relations')
        relations = {}
        for generator in relation_generators:
            relation = generator(dataframe)
            self._logger.info(f'loaded {relation.relation_type}')
            relations[relation.relation_type] = relation
        self.relation_set = TaFengRelationSet(**relations)
        self.relation_set.build_vocab()
