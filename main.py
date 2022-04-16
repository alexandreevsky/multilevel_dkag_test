import logging

import click

from data_loaders import load_amazon_dataframes, load_tafeng_dataframe
from knowledge_graph.graphs import AmazonGraph, TaFengGraph
from knowledge_graph.relations import get_amazon_relation_generators, get_tafeng_relation_generators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')


def get_amazon_graph(graph_name: str, user_k_core: int, item_k_core: int, data_folder_path: str = None) -> AmazonGraph:
    frames = load_amazon_dataframes(graph_name, user_k_core, item_k_core, data_folder_path)
    amazon_graph = AmazonGraph(frames, list(zip(
        ('reviews', 'meta', 'meta', 'meta', 'meta'),
        get_amazon_relation_generators()
    )))
    return amazon_graph


def get_tafeng_graph(user_k_core: int, item_k_core: int, data_folder_path: str = None) -> TaFengGraph:
    frame = load_tafeng_dataframe(user_k_core, item_k_core, data_folder_path)
    tafeng_graph = TaFengGraph(frame, get_tafeng_relation_generators())
    return tafeng_graph


@click.command()
@click.option("--graph_name", prompt='Amazon or TaFeng')
@click.option("--user_k_core", prompt='User k-core interactions graph')
@click.option("--item_k_core", prompt='Item k-core interactions graph')
def main(graph_name: str, user_k_core: int, item_k_core: int):
    graph = get_amazon_graph(graph_name, user_k_core, item_k_core) \
        if graph_name.lower() == 'amazon' else get_tafeng_graph(user_k_core, item_k_core)
    return graph


if __name__ == '__main__':
    main()
