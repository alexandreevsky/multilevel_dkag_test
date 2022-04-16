from collections import namedtuple
from typing import List, Tuple, Set, Union, Optional

import torch
from numpy import random
from torch import LongTensor
from torch.utils.data import Dataset

from knowledge_graph.types import AdjacencyLists, EdgeWithType

TransrPosNegBatch = namedtuple('PosNegBatch', 'head relation pos_tail neg_tail')
CustomerPosNegBatch = namedtuple('CustomerPosNegBatch', 'customer pos_products neg_products')
TimeStep = namedtuple('TimeBatch', 'time data')


class KgPosNegTriples(Dataset):
    def __init__(self, adj_lists: AdjacencyLists):
        self.adj_lists = adj_lists
        self.node_permutation = random.permutation(list(self.adj_lists.keys())).tolist()

    def __len__(self):
        return len(self.node_permutation)

    def sample_negative_triples_for_head(self, head: int, relation: int, n_samples: int) -> List[int]:
        pos_edges = list(filter(lambda x: x.relation_type == relation, self.adj_lists[head]))
        sampled_neg_tails = []
        while len(sampled_neg_tails) != n_samples:
            tail = random.choice(self.node_permutation, size=1).item()
            if EdgeWithType(relation_type=relation, to=tail) not in pos_edges and tail not in sampled_neg_tails:
                sampled_neg_tails.append(tail)
        return sampled_neg_tails

    def sample_positive_triples_for_head(self, head: int, n_samples: int) -> Tuple[List[int], List[int]]:
        pos_edges = self.adj_lists[head]
        if n_samples >= len(pos_edges):
            return list(map(lambda x: x.relation_type, pos_edges)), list(map(lambda x: x.to, pos_edges))
        tails = random.choice(pos_edges, size=n_samples, replace=False)
        return list(map(lambda x: x.relation_type, tails)), list(map(lambda x: x.to, tails))

    def _generate_triples(self, batch_heads: List[int]) -> Tuple[List[int], List[int], List[int]]:
        batch_positive_tails, batch_relations, batch_negative_tails = [], [], []
        for head in batch_heads:
            relations, positive_tails = self.sample_positive_triples_for_head(head, 1)
            batch_relations += relations
            batch_positive_tails += positive_tails
            batch_negative_tails += self.sample_negative_triples_for_head(head, batch_relations[-1], 1)
        return batch_relations, batch_positive_tails, batch_negative_tails

    def __getitem__(self, idx) -> Optional[
        Union[Tuple[List[int], List[int], List[int], List[int]], TransrPosNegBatch]
    ]:
        # ~ feature for TimeSplittedDataset ~
        # since subgraphs have different entity subsets we will pass None
        # if we try to get more items than the current dataset has
        if isinstance(idx, slice) and idx.start >= len(self):
            return None

        batch_heads = self.node_permutation[idx]
        if isinstance(idx, slice):
            batch_relations, batch_positive_tails, batch_negative_tails = self._generate_triples(batch_heads)
            return TransrPosNegBatch(
                head=batch_heads,
                relation=batch_relations,
                pos_tail=batch_positive_tails,
                neg_tail=batch_negative_tails
            )
        batch_relations, batch_positive_tails, batch_negative_tails = self._generate_triples([batch_heads])
        return batch_heads, batch_relations[0], batch_positive_tails[0], batch_negative_tails[0]


class KgCustomers(Dataset):
    def __init__(self,
                 splits: List[AdjacencyLists],
                 customer_indices: List[int],
                 product_indices: List[int],
                 purchase_relation_idx: int):
        self.splits = splits
        self.customers_to_iterate_over = None
        self.purchase_relation_idx = purchase_relation_idx
        self.product_indices = product_indices
        self.init(splits, set(customer_indices))

    def _customers_with_at_least_one_purchase(self,
                                              graph: AdjacencyLists,
                                              customer_indices: Set[int]) -> Set[int]:
        def has_at_least_one_purchase(customer) -> bool:
            return bool(len(list(
                filter(lambda edge: edge.relation_type == self.purchase_relation_idx, graph[customer])
            )))

        customer_generator = filter(lambda key: key in customer_indices, graph.keys())
        customers = set(filter(has_at_least_one_purchase, customer_generator))
        return customers

    def init(self,
             splits: List[AdjacencyLists],
             customer_indices: Set[int]) -> None:
        common_customers = self._customers_with_at_least_one_purchase(splits[-1], customer_indices)
        for idx in range(len(splits) - 1):
            train_split_customers = self._customers_with_at_least_one_purchase(splits[idx], customer_indices)
            common_customers &= train_split_customers
        self.customers_to_iterate_over = random.permutation(list(common_customers)).tolist()

    def sample_pos_products_for_customer(self,
                                         split_adj_lists: AdjacencyLists,
                                         customer_idx: int,
                                         n_samples: int) -> List[int]:
        pos_purchases = list(map(lambda x: x.to, filter(
            lambda x: x.relation_type == self.purchase_relation_idx,
            split_adj_lists[customer_idx]
        )))
        if n_samples >= len(pos_purchases):
            return pos_purchases

        return random.choice(pos_purchases, size=n_samples).tolist()

    def sample_neg_products_for_customer(self,
                                         split_adj_lists: AdjacencyLists,
                                         customer_idx: int,
                                         n_samples: int) -> List[int]:
        pos_purchases = list(map(lambda x: x.to, filter(
            lambda x: x.relation_type == self.purchase_relation_idx,
            split_adj_lists[customer_idx]
        )))

        neg_purchases = []
        while len(neg_purchases) != n_samples:
            product = random.choice(self.product_indices, size=1).item()
            if product not in pos_purchases and product not in neg_purchases:
                neg_purchases.append(product)
        return neg_purchases

    def __len__(self):
        return len(self.customers_to_iterate_over)

    def _generate_products(self, customer_idx: int) -> Tuple[int, LongTensor, LongTensor]:
        pos_purchases_through_time = []
        neg_purchases_through_time = []
        for split_idx in range(1, len(self.splits)):
            pos_purchases_through_time.append(
                self.sample_pos_products_for_customer(self.splits[split_idx], customer_idx, 1)[0]
            )
            neg_purchases_through_time.append(
                self.sample_neg_products_for_customer(self.splits[split_idx], customer_idx, 1)[0]
            )
        return customer_idx, \
               LongTensor(pos_purchases_through_time), \
               LongTensor(neg_purchases_through_time)

    def __getitem__(self, idx) -> Union[Tuple[List[int], LongTensor, LongTensor], CustomerPosNegBatch]:
        if isinstance(idx, slice):
            generated_tensors = list(map(
                lambda customer: self._generate_products(customer),
                self.customers_to_iterate_over[idx]
            ))
            return CustomerPosNegBatch(
                customer=list(map(lambda t: t[0], generated_tensors)),
                pos_products=torch.cat(tuple(map(lambda t: t[1].unsqueeze(0), generated_tensors))),
                neg_products=torch.cat(list(map(lambda t: t[2].unsqueeze(0), generated_tensors)))
            )
        return self._generate_products(self.customers_to_iterate_over[idx])


class TimeSplittedDataset(Dataset):
    def __init__(self, dataset_instances: List[Dataset]):
        self.dataset_instances = dataset_instances

    def __len__(self):
        return max(len(ds) for ds in self.dataset_instances)

    def __getitem__(self, idx) -> List[TimeStep]:
        # list of time steps with data == either None or list
        return list(map(lambda t: TimeStep(time=t[0], data=t[1][idx]), enumerate(self.dataset_instances)))
