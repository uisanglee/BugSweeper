import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import SAGEConv, GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import coalesce

from utils import tokens_to_indices
import numpy as np

import torch
import torch.nn.functional as F


class GraphClassifier(nn.Module):
	def __init__(self, embedding_dim, output_dim):
		super(GraphClassifier, self).__init__()
		self.fc1 = nn.Linear(embedding_dim, embedding_dim)
		self.dropout = nn.Dropout(0.3)
		self.fc2 = nn.Linear(embedding_dim, embedding_dim)
		self.dropout2 = nn.Dropout(0.3)
		self.out = nn.Linear(embedding_dim, output_dim)
		
	def forward(self, graph_embedding):
		x = self.fc1(graph_embedding)
		
		x = torch.relu(x)
		x = self.dropout(x)
		
		x = self.fc2(x)
		
		x = torch.relu(x)
		x = self.out(x)
		return x
		
class PoolEncoder(torch.nn.Module):
	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, vocab_size: int =0):
		super(PoolEncoder, self).__init__()
		
		#Embedding layer
		self.embed = nn.Embedding(vocab_size, in_channels)
		
		#Pooling layers
		
		self.convs = nn.ModuleList()
		self.convs.append(SAGEConv(in_channels, hidden_channels))
		
		for _ in range(num_layers - 1):
			self.convs.append(SAGEConv(hidden_channels, hidden_channels))
			
		self.relu = nn.ReLU()
		
	def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, pool: list) -> Tensor:
		
		mask = (x != 0).float()                 
		node_length_tensor = mask.sum(dim=1).clamp(min=1)	
			
		x_tensor = x
		node_embeddings = self.embed(x_tensor)
		
		flat_pool = []
		batch_idx = []
		offset = 0
		
		for big_graph_id, pool_sub in enumerate(pool):
			# 1) 이 그래프에 등장하는 sub-graph ID의 개수(n_sub)를 알아내고
			unique_subs = sorted(set(pool_sub))
			n_sub       = len(unique_subs)
			# 2) 로컬 ID → 글로벌 ID 매핑 생성
			mapping = { local_id: global_id
						for global_id, local_id in enumerate(unique_subs, start=offset) }
			# 3) 풀(flat) 리스트에 글로벌 ID를 채워넣고,
			flat_pool.extend(mapping[l] for l in pool_sub)
			#    batch 텐서용으로 어느 그래프 소속인지도 같이 기록
			batch_idx.extend([big_graph_id] * len(pool_sub))
			# 4) 다음 그래프의 글로벌 offset 을 늘려줌
			offset += n_sub
		
		# 이제 flat_pool, batch_idx 길이는 x.size(0) 과 동일
		pool_idx = torch.tensor(flat_pool, dtype=torch.long, device=x.device)
		batch = torch.tensor(batch_idx, dtype=torch.long, device=x.device)
							
		mask = (x_tensor != 0).float()
		sum_embeddings = (node_embeddings * mask.unsqueeze(-1)).sum(dim=1)
		average_node_embeddings = sum_embeddings / node_length_tensor.unsqueeze(-1)
		
		x = average_node_embeddings
		
		for conv in self.convs:
			x = conv(x, edge_index)
			x = self.relu(x)
			x = F.dropout(x, p=0.5, training=self.training)
		
		x_pooled = global_mean_pool(x, pool_idx)
		batch_pooled = global_max_pool(batch.unsqueeze(-1).float(), pool_idx)
		batch_pooled = batch_pooled.squeeze(1).long()
		
		src, dst = edge_index
		src_p = pool_idx[src]
		dst_p = pool_idx[dst]
		
		edge_index_pooled, _ = coalesce(
			torch.stack([src_p, dst_p], dim=0),
			None,
			src_p.max().item() + 1,
			src_p.max().item() + 1
		)
		
		mask = edge_index_pooled[0] != edge_index_pooled[1]
		edge_index_pooled = edge_index_pooled[:, mask]
		
		return x_pooled, edge_index_pooled, batch_pooled
		
class GNNModel(torch.nn.Module):
	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, pool_layer: int, num_layer: int ,vocab_size: int =0, heads: int = 4, final_concat: bool = False):
		"""
		Graph Neural Network Model with Pooling and GATConv layers.
		:param in_channels: Input feature dimension.
		:param
		hidden_channels: Hidden feature dimension.
		:param out_channels: Output feature dimension.
		:param pool_layer: Number of pooling layers.
		:param num_layer: Number of GATConv layers.
		:param vocab_size: Size of the vocabulary for embedding.
		:param heads: Number of attention heads for GATConv.
		:param final_concat: Whether to concatenate the output of the last GATConv layer.
		"""
		super(GNNModel, self).__init__()
		
		self.encoder = PoolEncoder(in_channels, hidden_channels, out_channels, pool_layer, vocab_size)
		self.convs = nn.ModuleList()
		
		self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
		conv_output_dim = hidden_channels * heads
		
		for _ in range(num_layer - 1):
			self.convs.append(GATConv(conv_output_dim, hidden_channels, heads=1, concat=True))
			conv_output_dim = hidden_channels
		
		self.classifier = GraphClassifier(hidden_channels, out_channels)

	def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, pool: list) -> Tensor:
		x_pooled, edge_index_pooled, batch_pooled = self.encoder(x, edge_index, batch, pool)
		
		for conv in self.convs:
			x_pooled = conv(x_pooled, edge_index_pooled)
			x_pooled = F.relu(x_pooled)
			x_pooled = F.dropout(x_pooled, p=0.5, training=self.training) 
			
		graph_pooled = global_mean_pool(x_pooled, batch_pooled)
		out = self.classifier(graph_pooled)
		return out