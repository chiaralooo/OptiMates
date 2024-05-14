import csv
import logging
import math
from itertools import product
from typing import Any, Iterable

import motile
import networkx as nx
import numpy as np
from LineageTree import lineageTree
from scipy.spatial import KDTree
from skimage.measure import regionprops
from tqdm import tqdm


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: None | dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")
    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["t"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> KDTree:
    positions = [cand_graph.nodes[node]["pos"] for node in node_ids]
    return KDTree(positions)


def to_motile(lT: lineageTree, crop: int = None, max_dist=200):
    fmt = nx.DiGraph()
    if not crop:
        crop = lT.t_e
    # time_nodes = [
    for time in range(crop):
        #     time_nodes += lT.time_nodes[time]
        # print(time_nodes)
        for time_node in lT.time_nodes[time]:
            fmt.add_node(
                time_node,
                **{"t": lT.time[time_node], "pos": lT.pos[time_node], "score": 1},
            )
            # for suc in lT.successor:
            #     fmt.add_edge(time_node, suc, **{"score":0})
        add_cand_edges(fmt, max_dist)

    return fmt


def write_csv_from_lT_to_lineaja(lT, path_to, start: int = 200, finish: int = 300):
    csv_dict = {}
    for time in range(start, finish):
        for node in lT.time_nodes[time]:
            csv_dict[node] = {"pos": lT.pos[node], "t": time}
    with open(path_to, "w", newline="\n") as file:
        fieldnames = ["time", "positions_x", "positions_y", "positions_z", "id"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for node in csv_dict.keys():
            writer.writerow(
                {
                    "time": csv_dict[node]["t"],
                    "positions_x": csv_dict[node]["pos"][0],
                    "positions_y": csv_dict[node]["pos"][1],
                    "positions_z": csv_dict[node]["pos"][2],
                    "id": node,
                }
            )
