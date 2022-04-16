import logging
import random

import numpy as np
import torch
import torch.optim as opt
from toolz import valmap
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from knowledge_graph.datasets import KgPosNegTriples, TimeSplittedDataset, KgCustomers
from knowledge_graph.graphs import KnowledgeGraph
from knowledge_graph.layer_generators import LayerNodeGenerator
from metrics import calculate_metrics_at_k
from models.Model import Model
from models.config import Config
from utils import get_dates_for_split, get_graph_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
_logger = logging.getLogger(__file__)


def seed_random(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_transR_one_epoch(
        model: Model,
        dataset: Dataset,
        optimizer: Optimizer,
        batch_size: int,
        verbose: int,
        use_tqdm: bool = False) -> None:
    batch_idx = 0
    n_iters = len(dataset) // batch_size + 1
    total_loss = 0
    iterable = tqdm(range(1, n_iters + 1)) if use_tqdm else range(1, n_iters + 1)
    for iter_idx in iterable:
        batches = dataset[batch_idx: batch_idx + batch_size]
        batches = list(map(lambda x: x.data, batches))

        inputs = list(map(
            lambda x: None if x is None else list(set(x.head + x.pos_tail + x.neg_tail)), batches
        ))

        time_outputs = model(inputs, transR=True)

        batches = list(filter(lambda x: x is not None, batches))
        time_outputs = list(filter(lambda x: x is not None, time_outputs))

        loss = model.transR_loss(batches, time_outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if iter_idx % verbose == 0:
            _logger.info(f'Iter {iter_idx}: batch loss -> {loss.item()}, mean loss -> {total_loss / iter_idx}')
        batch_idx += batch_size


def train_lstm_one_epoch(model: Model,
                         dataset: Dataset,
                         optimizer: Optimizer,
                         batch_size: int,
                         verbose: int,
                         use_tqdm: bool = False) -> None:
    batch_idx = 0
    n_iters = len(dataset) // batch_size + 1
    total_loss = 0
    iterable = tqdm(range(1, n_iters + 1)) if use_tqdm else range(1, n_iters + 1)
    for iter_idx in iterable:
        batch = dataset[batch_idx: batch_idx + batch_size]
        loss = model.recommender_task_loss(batch)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_idx % verbose == 0:
            _logger.info(f'Iter {iter_idx}: batch loss -> {loss.item()}, mean loss -> {total_loss / iter_idx}')
        batch_idx += batch_size


def train(model: Model,
          agg_optimizer: Optimizer,
          lstm_optimizer: Optimizer,
          kg_ds: Dataset,
          customer_ds: Dataset,
          n_epochs: int,
          batch_size: int,
          verbose: int) -> None:
    for epoch_idx in range(1, n_epochs + 1):
        _logger.info(f'Epoch #{epoch_idx}:')
        _logger.info('Fitting TransR:')
        train_transR_one_epoch(model, kg_ds, agg_optimizer, batch_size, verbose)
        _logger.info('Fitting Lstm:')
        train_lstm_one_epoch(model, customer_ds, lstm_optimizer, batch_size, verbose)
    _logger.info('Finished training')


def evaluate(model: Model, interactions_dict: dict, product_ids: list, batch_size: int, k: int, use_tqdm: bool) -> tuple:
    customer_ids = list(interactions_dict.keys())

    model.eval()

    precision_k = []
    recall_k = []
    ndcg_k = []

    with torch.no_grad():
        product_embeddings = model(product_ids, transR=False)

        product_to_idx = {product: idx for idx, product in enumerate(product_ids)}

        interactions_dict = valmap(
            lambda x_list: list(map(lambda item: product_to_idx[item], x_list)),
            interactions_dict
        )

        batch_idx = 0
        n_batches = len(customer_ids) // batch_size + 1
        iterable = tqdm(range(n_batches)) if use_tqdm else range(n_batches)
        for _ in iterable:
            customer_batch = customer_ids[batch_idx: batch_idx + batch_size]
            interactions_list = [interactions_dict[customer] for customer in customer_batch]

            scores = model(customer_batch, transR=False) @ product_embeddings.T
            batch_precision, batch_recall, batch_ndcg = calculate_metrics_at_k(
                k, scores.cpu(), interactions_list
            )

            precision_k.append(batch_precision)
            recall_k.append(batch_recall)
            ndcg_k.append(batch_ndcg)

            batch_idx += batch_size

    precision_k = sum(np.concatenate(precision_k)) / len(customer_ids)
    recall_k = sum(np.concatenate(recall_k)) / len(customer_ids)
    ndcg_k = sum(np.concatenate(ndcg_k)) / len(customer_ids)

    return precision_k, recall_k, ndcg_k


def train(n_epochs: int, n_splitting_points: int, batch_size: int, knowledge_graph: KnowledgeGraph, seed: int) -> None:
    seed_random(seed)

    timestamps = knowledge_graph.relation_set.get_all_timestamps()
    splitting_points = get_dates_for_split(timestamps, n_points=n_splitting_points)

    graph_splits = get_graph_splits(knowledge_graph, splitting_points)

    test_split = graph_splits[-1]
    train_splits = graph_splits[:-1]

    config = Config(
        n_timesplits=len(train_splits),
        entity_embedding_dim=10,
        relation_embedding_dim=10,
        n_entities=len(knowledge_graph.entity_set),
        n_relations=len(knowledge_graph.relation_set),
        n_layers=2,
        lstm_hidden_size=10,
        transR_l2_weight=0.05
    )
    model = Model(config)

    agg_optimizer = opt.Adam(model.transR_aggregator.parameters())
    lstm_optimizer = opt.Adam(model.lstm.parameters())

    layer_generators = [LayerNodeGenerator(split, n_neighbours=8, n_layers=config.n_layers)
                        for split in train_splits]
    pos_neg_triples = [KgPosNegTriples(split) for split in train_splits]
    time_wrapped_pos_neg_triples = TimeSplittedDataset(pos_neg_triples)

    customer_indices = list(
        map(lambda x: knowledge_graph.entity_set.entity2idx[x], knowledge_graph.entity_set.customer)
    )
    product_indices = list(map(
        lambda x: knowledge_graph.entity_set.entity2idx[x], knowledge_graph.entity_set.product
    ))

    customer_ds = KgCustomers(
        splits=train_splits,
        customer_indices=customer_indices,
        product_indices=product_indices,
        purchase_relation_idx=knowledge_graph.relation_set.relation2idx['purchase']
    )

    for epoch_idx in range(1, n_epochs + 1):
        _logger.info(f'Epoch #{epoch_idx}:')
        train_transR_one_epoch(
            model=model,
            dataset=time_wrapped_pos_neg_triples,
            layer_generators=layer_generators,
            optimizer=agg_optimizer,
            batch_size=batch_size,
            verbose=10
        )
        train_lstm_one_epoch(
            model=model,
            dataset=customer_ds,
            layer_generators=layer_generators,
            optimizer=lstm_optimizer,
            batch_size=batch_size,
            verbose=10
        )


if __name__ == '__main__':
    pass
