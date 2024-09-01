import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from graph.util.time import *
from graph.util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


import itertools
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
import random

# Helper function to create a directed graph from the adjacency matrix
def create_graph_from_adjacency_matrix(adj_matrix):
    G = nx.DiGraph()
    for i, targets in enumerate(adj_matrix):
        for j in targets:
            if i != j:  # Avoid self-loops
                G.add_edge(i, j.item())
    return G

# Function to flip two communities in a directed graph
def flip_two_communities_directed(adj_matrix, communities, community_id1, community_id2):
    if community_id1 not in communities or community_id2 not in communities:
        raise ValueError("One or both community IDs do not exist.")
    
    flipped_adj_matrix = np.copy(adj_matrix)
    nodes1 = communities[community_id1]
    nodes2 = communities[community_id2]
    
    node_mapping = {node: node for node in range(len(adj_matrix))}
    for n1, n2 in zip(nodes1, nodes2):
        node_mapping[n1] = n2
        node_mapping[n2] = n1
    
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            flipped_adj_matrix[node_mapping[i], node_mapping[j]] = adj_matrix[i, j]
    
    return flipped_adj_matrix

def adjacency_matrix_to_list(flipped_adj_matrix, original_format):
    adjacency_list = []
    for i, row in enumerate(flipped_adj_matrix):
        connected_nodes = np.nonzero(row)[0]
        # If the original format includes self-references, adjust accordingly; otherwise, filter them out
        filtered_nodes = [node for node in connected_nodes if node != i][:len(original_format[i])]
        filtered_nodes.insert(0, i)
        adjacency_list.append(filtered_nodes)
    return adjacency_list


# Detect communities using the Girvan-Newman algorithm
def detect_communities_girvan_newman(G, target_communities=4):
    G_undirected = G.to_undirected()
    comm_iter = girvan_newman(G_undirected)
    for communities in itertools.islice(comm_iter, target_communities-1):
        if len(communities) >= target_communities:
            return communities


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        print(in_channel)
        print(out_channel)
        print(inter_dim)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, Mydevice = 'cuda'):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        #device = get_device()
        device = Mydevice

        edge_index = edge_index_sets[0]


        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        #
        encoder_dim = dim
        self.encoder = nn.Linear(node_num * dim, encoder_dim)
        pred_dim = 6 + 1
        self.arrangement_pred_layer = nn.Linear(encoder_dim, pred_dim)
        #

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        # 
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.arrangement_pred_layer.weight, a=math.sqrt(5))
        #


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()


        gcn_outs = []
        selected_combination_arr = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            ### Puzzle Making
            
            X1 = topk_indices_ji.cpu().numpy()  
            G = create_graph_from_adjacency_matrix(X1)

            # Target number of communities
            target_communities = 4
            partition = detect_communities_girvan_newman(G, target_communities)

            # Convert partition to node_to_community format
            node_to_community = {}
            for idx, community in enumerate(partition):
                for node in community:
                    node_to_community[node] = idx

            # Organize communities
            communities = {}
            for node, community_id in node_to_community.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

            possible_num = 6
            selected_combination = random.randint(0, possible_num)
            if selected_combination == 0:
                sorted_combination = [0, 1]
            elif selected_combination == 1:
                sorted_combination = [0, 2]
            elif selected_combination == 2:
                sorted_combination = [0, 3]
            elif selected_combination == 3:
                sorted_combination = [1, 2]
            elif selected_combination == 4:
                sorted_combination = [1, 3]
            elif selected_combination == 5:
                sorted_combination = [2, 3]
            else:
                sorted_combination = None
            if( selected_combination != possible_num ):
              community_id1 = sorted_combination[0]
              community_id2 = sorted_combination[1]


              # Flip communities
              adj_matrix_directed = np.zeros((len(X1),len(X1)))
              for ii, targets in enumerate(X1):
                for jj in targets:
                  if ii != jj:  # Avoid self-loops
                    adj_matrix_directed[ii,jj] = 1

              flipped_adj_matrix_directed = flip_two_communities_directed(adj_matrix_directed, communities, community_id1, community_id2)
              
  
              # Convert the flipped graph back to the adjacency list format
              flipped_adj_list = adjacency_matrix_to_list(flipped_adj_matrix_directed, topk_indices_ji.cpu().numpy())
  
              # Convert the adjacency list to a tensor matching the original X's device and data type
              output = torch.tensor(flipped_adj_list, dtype=topk_indices_ji.dtype).to(topk_indices_ji.device)
              
              topk_indices_ji = output

            
            

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        for i in range(batch_num):
          selected_combination_arr.append(selected_combination)

        x_flattened = x.view(batch_num, -1)
        encoded_latent_space = self.encoder(x_flattened)
        encoded_latent_space_flattened = encoded_latent_space.view(batch_num, -1)
        arrangement_predictions = self.arrangement_pred_layer(encoded_latent_space_flattened)

        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        
        return encoded_latent_space, out, arrangement_predictions, torch.tensor(selected_combination_arr).to(device)
        