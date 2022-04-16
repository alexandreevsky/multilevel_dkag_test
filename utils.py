from datetime import datetime
from typing import List, Dict

from toolz import valmap, valfilter, keyfilter

from data_loaders import load_amazon_dataframes, load_tafeng_dataframe
from knowledge_graph.graphs import AmazonGraph, TaFengGraph, KnowledgeGraph
from knowledge_graph.relations import get_amazon_relation_generators, get_tafeng_relation_generators
from knowledge_graph.types import AdjacencyLists
from models.Model import Model
import torch


def get_test_interactions(
        customer_ids: List[int],
        test_graph_split: AdjacencyLists,
        purchase_relation_ids: int) -> Dict[int, List[int]]:
    # to speed up search operations
    customer_ids = set(customer_ids)

    customer_interactions = valmap(
        lambda edges: filter(lambda edge: edge.relation_type == purchase_relation_ids, edges),
        keyfilter(lambda key: key in customer_ids, test_graph_split)
    )
    return valfilter(
        len,
        valmap(lambda edges: list(set(map(lambda edge: edge.to, edges))), customer_interactions)
    )


def get_dates_for_split(timestamps, n_points: int) -> List[datetime]:
    idx, step, dates = 0, len(timestamps) // (n_points + 1), []
    while len(dates) < n_points:
        idx += step
        dates.append(datetime.fromtimestamp(timestamps[idx]))
    return dates


def get_graph_splits(knowledge_graph: KnowledgeGraph, splitting_points: List[datetime]) -> List[AdjacencyLists]:
    splits = []
    for idx in range(len(splitting_points) + 1):
        if not idx:
            adj_lists = knowledge_graph.to_adj_lists(before=splitting_points[idx])
        elif idx == len(splitting_points):
            adj_lists = knowledge_graph.to_adj_lists(after=splitting_points[idx - 1])
        else:
            adj_lists = knowledge_graph.to_adj_lists(after=splitting_points[idx - 1], before=splitting_points[idx])
        splits.append(adj_lists)
    return splits


def get_amazon_graph(graph_name: str, user_k_core: int, item_k_core: int) -> AmazonGraph:
    frames = load_amazon_dataframes(graph_name, user_k_core, item_k_core)
    amazon_graph = AmazonGraph(frames, list(zip(
        ('reviews', 'meta', 'meta', 'meta', 'meta'),
        get_amazon_relation_generators()
    )))
    return amazon_graph


def get_tafeng_graph(user_k_core: int, item_k_core: int) -> TaFengGraph:
    frame = load_tafeng_dataframe(user_k_core, item_k_core)
    tafeng_graph = TaFengGraph(frame, get_tafeng_relation_generators())
    return tafeng_graph


def load_weights(model: Model, transR_checkpoint_path: str, lstm_checkpoint_path: str) -> None:
    checkpoint = torch.load(transR_checkpoint_path)
    model.transR_aggregator.load_state_dict(checkpoint['transR_aggregator'])
    checkpoint = torch.load(lstm_checkpoint_path)
    model.lstm.load_state_dict(checkpoint['lstm'])
