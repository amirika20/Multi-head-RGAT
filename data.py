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

# for building training/testing graphs
def get_subset_g(g, mask, num_rels, bidirected=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata["etype"][mask]

    if bidirected:
        sub_src, sub_dst = torch.cat([sub_src, sub_dst]), torch.cat(
            [sub_dst, sub_src]
        )
        sub_rel = torch.cat([sub_rel, sub_rel + num_rels])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata[dgl.ETYPE] = sub_rel
    return sub_g

class GlobalUniform:
    def __init__(self, g, sample_size):
        self.sample_size = sample_size
        self.eids = np.arange(g.num_edges())

    def sample(self):
        return torch.from_numpy(np.random.choice(self.eids, self.sample_size))


class NegativeSampler:
    def __init__(self, k=10):  # negative sampling rate = 10
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values = np.random.randint(num_nodes, size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        samples = np.concatenate((pos_samples, neg_samples))

        # binary labels indicating positive and negative samples
        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1

        return torch.from_numpy(samples), torch.from_numpy(labels)


class SubgraphIterator:
    def __init__(self, g, num_rels, sample_size=30000, num_epochs=6000):
        self.g = g
        self.num_rels = num_rels
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.pos_sampler = GlobalUniform(g, sample_size)
        self.neg_sampler = NegativeSampler()

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        # relabel nodes to have consecutive node IDs
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        # edges is the concatenation of src, dst with relabeled ID
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)

        # use only half of the positive edges
        chosen_ids = np.random.choice(
            np.arange(self.sample_size),
            size=int(self.sample_size / 2),
            replace=False,
        )
        src = src[chosen_ids]
        dst = dst[chosen_ids]
        rel = rel[chosen_ids]
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_rels))
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = torch.from_numpy(rel)
        sub_g.edata["norm"] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = torch.from_numpy(uniq_v).view(-1).long()

        return sub_g, uniq_v, samples, labels
    
    
