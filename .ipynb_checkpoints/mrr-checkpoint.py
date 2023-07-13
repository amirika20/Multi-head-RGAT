import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.sampling import global_uniform_negative_sampling
from dgl.dataloading import GraphDataLoader

def filter(
    triplets_to_filter, target_s, target_r, target_o, num_nodes, filter_o=True
):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]
    for e in range(num_nodes):
        triplet = (
            (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        )
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter:
            candidate_nodes.append(e)
    return torch.LongTensor(candidate_nodes)


def perturb_and_get_filtered_rank(
    model, node_embedding, edge_embedding, s, r, o, test_size, triplets_to_filter, filter_o=True
):
    """Perturb subject or object in the triplets"""
    num_nodes = node_embedding.shape[1]
    ranks = []
    for idx in tqdm.tqdm(range(test_size), desc="Evaluate"):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(
            triplets_to_filter,
            target_s,
            target_r,
            target_o,
            num_nodes,
            filter_o=filter_o,
        )
        num_triplets = candidate_nodes.shape[0]
        if filter_o:
            triplets = torch.cat([target_s.expand((num_triplets,1)),target_r.expand((num_triplets,1)), candidate_nodes],dim=1)
        else:
            triplets = torch.cat([candidate_nodes,target_r.expand((num_triplets,1)), target_o.expand((num_triplets,1))],dim=1)
        target_idx = 0
        scores = model.calc_score(triplets, node_embedding, edge_embedding)

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_mrr(
    model, node_embedding, edge_embedding, test_mask, triplets_to_filter, batch_size=100, filter=True
):
    with torch.no_grad():
        test_triplets = triplets_to_filter[test_mask]
        s, r, o = test_triplets[:, 0], test_triplets[:, 1], test_triplets[:, 2]
        test_size = len(s)
        triplets_to_filter = {
            tuple(triplet) for triplet in triplets_to_filter.tolist()
        }
        ranks_s = perturb_and_get_filtered_rank(
            model, node_embedding, edge_embedding, s, r, o, test_size, triplets_to_filter, filter_o=False
        )
        ranks_o = perturb_and_get_filtered_rank(
            model, node_embedding, edge_embedding, s, r, o, test_size, triplets_to_filter
        )
        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed
        mrr = torch.mean(1.0 / ranks.float()).item()
    return mrr