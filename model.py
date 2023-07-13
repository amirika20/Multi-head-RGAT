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

class MultiHeadLinearTransformer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int, bias=True):
        super(MultiHeadLinearTransformer, self).__init__()
        self.weight = nn.Parameter(torch.empty((num_heads, out_features, in_features)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight, gain=1)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias, gain=1)

    def forward(self, x):
        num_entities = x.shape[1]
        y = torch.matmul(x, torch.transpose(self.weight,-1,-2))
        if self.bias is not None:
            y += self.bias.unsqueeze(1).repeat(1,num_entities,1)
        return y
    
class MultiHeadRelationLinearTransformer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_rels : int, num_heads: int, bias=True):
        super(MultiHeadRelationLinearTransformer, self).__init__()
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.empty((num_heads, num_rels, out_features, in_features)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, num_rels, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight, gain=1)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias, gain=1)

    def forward(self, x):
        assert x.shape[0]==self.num_heads, f'Expected {self.num_heads} number of heads, but input has {x.shape[0]} number of heads'
        num_entities = x.shape[1]
        y = torch.matmul(x.unsqueeze(dim=2), torch.transpose(self.weight,-1,-2)).squeeze(dim=(2))
        if self.bias is not None:
            y += self.bias
        return y
    
    
class MultiHeadEdgeAttention(nn.Module):
    def __init__(self, in_features: int, num_heads: int, bias= True, activation = 'leaky relu'):
        super(MultiHeadEdgeAttention, self).__init__()
        self.weight = nn.Parameter(torch.empty((num_heads, 1, 3*in_features)))
        self.bias = None
        self.num_heads = num_heads
        self.in_features = in_features

        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, 1))
        if activation=='leaky relu':
            self.activation = F.leaky_relu
        elif activation=='relu':
            self.activation = F.relu
        else:
          raise Exception(f'{activation} is not a vlid activation')

        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias, gain=gain)

    def forward(self, src, edge, dst):
        assert src.shape[0]==self.num_heads, f'Expected {self.num_heads} heads, but src has {src.shape[0]} heads'
        assert edge.shape[0]==self.num_heads, f'Expected {self.num_heads} heads, but edge has {edge.shape[0]} heads'
        assert dst.shape[0]==self.num_heads, f'Expected {self.num_heads} heads, but dst has {dst.shape[0]} heads'
        assert src.shape[2]==self.in_features, f'Expected {self.in_features} features, but src has {src.shape[2]} features'
        assert edge.shape[2]==self.in_features, f'Expected {self.in_features} features, but edge has {edge.shape[2]} features'
        assert dst.shape[2]==self.in_features, f'Expected {self.in_features} features, but dst has {dst.shape[2]} features'

        z = torch.cat([src, edge, dst], dim=2)
        attention = torch.bmm(z, torch.transpose(self.weight,-1,-2))
        num_edges = edge.shape[1]
        if self.bias is not None:
            attention += self.bias.unsqueeze(1).repeat(1,num_edges,1)
        return self.activation(attention)
    

class MultiHeadGATSingleLayer(nn.Module):
    def __init__(self, node_dim : int, edge_dim : int, out_dim :int, num_rels:int, num_heads:int, cat=True, feat_drop = 0.0, attn_drop = 0, bias=True, activation = 'leaky relu', self_loop = True):
        super(MultiHeadGATSingleLayer, self).__init__()
        # Parameters
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert out_dim%num_heads==0, 'Expected output dim to be a intiger multiplication of number of heads'
        self.head_dim = int(out_dim/num_heads)

        # If self loop is required, we define a new edge type specific for self loop
        self.self_loop = self_loop
        if self.self_loop:
            self.self_loop_w = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
            nn.init.xavier_normal_(self.self_loop_w, gain=1)
        self.num_rels = num_rels
        self.bias = bias
        self.cat = cat
        self.activation = None
        if activation=='leaky relu':
            self.activation = F.leaky_relu
        elif activation=='relu':
            self.activation = F.relu
        else:
          raise Exception(f'{activation} is not a vlid activation')

        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Node projection
        self.node_prj = MultiHeadLinearTransformer(node_dim, self.head_dim, num_heads)

        # Edge projection
        self.edge_prj = MultiHeadRelationLinearTransformer(edge_dim, self.head_dim, self.num_rels, num_heads)

        # Attention
        self.attn_fc = MultiHeadEdgeAttention(self.head_dim, num_heads, bias, activation)

    def edge_attention(self, edges):
        """
        Computes attention factor for each edge
        """
        # Attention factor is based on the embedding of source nodes, edges, and the destination nodes
        src = edges.src["embedding"].permute(1,0,2)
        edge = edges.data['embedding'].permute(1,0,2)
        dst = edges.dst["embedding"].permute(1,0,2)
        return {"attention": self.attn_fc(src,edge,dst).permute(1,0,2)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        
        return {"src_data": edges.src["embedding"], "edge_data": edges.data['embedding'], "attention" : edges.data["attention"]}

    def reduce_func(self, nodes):
        alpha = self.attn_drop(F.softmax(nodes.mailbox["attention"].permute(0,2,1,3), dim=2))
        new_embedding = torch.sum(alpha * nodes.mailbox["src_data"].permute(0,2,1,3) * nodes.mailbox["edge_data"].permute(0,2,1,3), dim=2) 
        
        return {"agg_msg": new_embedding}

    def apply_func(self, nodes):
        new_embedding = nodes.data['agg_msg']
        if self.self_loop:
            new_embedding += torch.matmul(nodes.data['embedding'].permute(1,0,2), torch.transpose(self.self_loop_w, -1, -2)).permute(1,0,2)
        return {'out': self.activation(new_embedding)}

    def forward(self, g, node_feat, edge_feat):
        """
        g.ndata must have dgl.NTYPE
        g.edata must have dgl.ETYPE
        node_feat : number of entities x node_feat dim
        edge_feat : number of relations x edge_feat dim
        """
        assert edge_feat.shape[0]==self.num_rels, f'Expected the {self.num_rels} relation embedding, but {edge_feat.shape[0]} is given'
        assert edge_feat.shape[1]==self.edge_dim, f'Expected the dimension of edge embedding to be {self.edge_dim}, but {edge_feat.shape[1]} is given'
        assert g.num_nodes()==node_feat.shape[0], f'Expected {g.num_nodes()} of node embedding, but {node_feat.shape[0]} is given'

        num_nodes = g.num_nodes()
        num_edges = g.num_edges()

        with g.local_scope():
            eids = g.edata[dgl.ETYPE]

            # Coping the same embedding to pass to all the heads
            node_feat = node_feat.repeat(self.num_heads, 1, 1)
            edge_feat = edge_feat.repeat(self.num_heads, 1, 1)
            # Projecting node features
            node_embedding = self.node_prj(node_feat)
            g.ndata["embedding"] = self.feat_drop(node_embedding.permute(1,0,2))

            # Projecting edge feature
            edge_embedding = self.edge_prj(edge_feat)
            g.edata["embedding"] = self.feat_drop(edge_embedding.permute(1,0,2)[eids])

            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func, self.apply_func)
            if self.cat:
                node_embedding = torch.cat(g.ndata["out"].permute(1,0,2).unbind(dim=0),dim=1)
                edge_embedding = torch.cat(edge_embedding.unbind(dim=0),dim=1)
            else:
                node_embedding = g.ndata["out"].permute(1,0,2)

            return node_embedding, edge_embedding
        
        
class RGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, num_rels, num_heads, num_layers, cat=False, feat_drop = 0.0, attn_drop = 0.0, bias=True, activation = 'leaky relu', self_loop = True):
        super(RGAT, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cat = cat
        self.activation = activation
        self.bias = bias
        self.self_loop = self_loop
        self.layers = nn.ModuleList()
        # input to hidden
        self.layers.append(MultiHeadGATSingleLayer(node_dim, edge_dim, out_dim, num_rels, num_heads, cat=True, feat_drop = feat_drop, attn_drop = attn_drop, bias=bias, activation = activation, self_loop = self_loop))
        # hidden to hidden
        for _ in range(num_layers-2):
            self.layers.append(MultiHeadGATSingleLayer(out_dim, out_dim, out_dim, num_rels, num_heads, cat=True, feat_drop = feat_drop, attn_drop = attn_drop, bias=bias, activation = activation, self_loop = self_loop))
        self.layers.append(MultiHeadGATSingleLayer(out_dim, out_dim, out_dim, num_rels, num_heads, cat=cat, feat_drop = feat_drop, attn_drop = attn_drop, bias=bias, activation = activation, self_loop = self_loop))

    def forward(self, g, node_feat, edge_feat):
        for layer in self.layers:
            node_feat, edge_feat = layer(g, node_feat, edge_feat)

        return node_feat, edge_feat
    

class Qatt(nn.Module):
    def __init__(self, node_dim: int, rel_dim: int, query_dim: int, num_heads: int):
      super(Qatt, self).__init__()
      self.node_dim = node_dim
      self.rel_dim = rel_dim
      self.q_dim = query_dim
      self.num_heads = num_heads

      self.sub_q = nn.Linear(int(node_dim/num_heads), query_dim)
      self.rel_q = nn.Linear(rel_dim, query_dim)
      self.attn = nn.Linear(int(node_dim/num_heads)+rel_dim, query_dim)
      self.scoring_fnc = nn.Linear(query_dim*num_heads, node_dim, bias = False)

    def forward(self, triplets, node_feat, edge_feat):

        q_num = triplets.shape[0]

        s = node_feat[:,triplets[:,0]]
        r = torch.cat(edge_feat[:,triplets[:,1]].unbind(dim=0), dim=1)
        o = torch.cat(node_feat[:,triplets[:,2]].unbind(dim=0), dim=1)

        s_q = self.sub_q(s)
        r_q = self.rel_q(r).unsqueeze(0)
        beta = torch.softmax(torch.sum(s_q*r_q,dim=2) / self.q_dim**0.5, dim=0).unsqueeze(2)

        z = torch.stack([torch.cat([s_head,r], dim=1) for s_head in s.unbind(dim=0)],dim=0)

        attn = self.attn(z) * beta
        Q = torch.cat(attn.unbind(dim=0),dim=1)
        score = F.leaky_relu(self.scoring_fnc(Q))
        score = torch.sum(score * o,dim=1)
        return F.sigmoid(score)
    
    
    
class LinkPredict(nn.Module):
    def __init__(self, num_nodes, num_rels, node_embedding_dim, edge_embedding_dim, hidden_dim, query_dim, num_heads, num_layers, reg_param=0.01, self_loop = True):
        super().__init__()
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.node_feat = nn.Embedding(num_nodes, node_embedding_dim)
        self.edge_feat = nn.Embedding(self.num_rels, edge_embedding_dim)
        self.rgat = RGAT(node_embedding_dim, edge_embedding_dim, hidden_dim, num_rels, num_heads, num_layers, self_loop=self_loop)
        self.qatt = Qatt(hidden_dim, hidden_dim, query_dim, num_heads)

    def forward(self, g, nids, triplets, device):
        eids = torch.arange(self.num_rels).to(device)
        node_embedding, edge_embedding = self.rgat(g, self.node_feat(nids), self.edge_feat(eids))
        return self.qatt(triplets, node_embedding, edge_embedding)

    def get_loss(self, score, labels):
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def get_embedding(self, g, nids):
        with torch.no_grad():
            return self.rgat(g, self.node_feat(nids), self.edge_feat.weight)


    def calc_score(self, triplets, node_embedding, edge_embedding):
        with torch.no_grad():
            return self.qatt(triplets, node_embedding, edge_embedding)

