import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import pickle
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.sampling import global_uniform_negative_sampling
from dgl.dataloading import GraphDataLoader
from data import *
from model import *
from mrr import *

def train(
    dataloader,
    test_g,
    test_nids,
    test_mask,
    triplets,
    device,
    model_state_file,
    model,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    best_mrr = 0
    for epoch, batch_data in enumerate(dataloader):  # single graph batch
        model.train()
        g, train_nids, edges, labels = batch_data
        g = g.to(device)
        g.ndata[dgl.NTYPE] = train_nids.to(device)
        train_nids = train_nids.to(device)
        edges = edges.to(device)
        labels = labels.to(device)
        

        scores = model(g, train_nids, edges, device)
        loss = model.get_loss(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # clip gradients
        optimizer.step()
        scheduler.step()
        print(
            "Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f}".format(
                epoch, loss.item(), best_mrr
            )
        )
        if (epoch + 1) % 100 == 0:
            # perform validation on CPU because full graph is too large
            device = torch.device('cpu')
            model = model.cpu()
            model.eval()
            test_rel_types = test_g.edata['_TYPE']
            node_embedding, edge_embedding = model.get_embedding(test_g, test_nids)
            mrr = calc_mrr(
                model, node_embedding, edge_embedding, test_mask, triplets, batch_size=100
            )
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch},
                    model_state_file,
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with DGL built-in RGCN module")
    # load and preprocess dataset
    with open('data.pickle', 'rb') as handle:
        data = pickle.load(handle)
    g = data[0]
    num_nodes = g.num_nodes()
    num_rels = data.num_rels
    train_g = get_subset_g(g, g.edata["train_mask"], num_rels)
    test_g = get_subset_g(g, g.edata["train_mask"], num_rels, bidirected=True)
    test_nids = torch.arange(0, num_nodes)
    test_mask = g.edata["test_mask"]
    subg_iter = SubgraphIterator(train_g, num_rels)  # uniform edge sampling
    dataloader = GraphDataLoader(
        subg_iter, batch_size=1, collate_fn=lambda x: x[0]
    )

    # Prepare data for metric computation
    src, dst = g.edges()
    triplets = torch.stack([src, g.edata["etype"], dst], dim=1)

    # create RGCN model
    model = LinkPredict(14541, 474, 128, 128, 128, 128, 8, 3, self_loop=True).to(device)
    model.to(device)
    # train
    model_state_file = "model_state.pth"
    train(
        dataloader,
        test_g,
        test_nids,
        test_mask,
        triplets,
        device,
        model_state_file,
        model,
    )

    # testing
    print("Testing...")
    checkpoint = torch.load(model_state_file)
    model = model.cpu()  # test on CPU
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])
    embed = model(test_g, test_nids)
    best_mrr = calc_mrr(
        embed, model.w_relation, test_mask, triplets, batch_size=500
    )
    print(
        "Best MRR {:.4f} achieved using the epoch {:04d}".format(
            best_mrr, checkpoint["epoch"]
        )
    )
