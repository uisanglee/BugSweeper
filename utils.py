# General
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Pytorch
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn import Sigmoid
import torch.nn as nn
from torch_geometric.data import Dataset, Data

# Matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Sci-kit Learn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

from torch_geometric.data import InMemoryDataset, Dataset
import pickle
import os

from tokenizers import ByteLevelBPETokenizer

# Global variables

def tokens_to_indices(encoded_batch, tokenizer):
	vocab = tokenizer.get_vocab()
	
	return [[[vocab[token] for token in encoding.tokens]for encoding in encodings ] for encodings in encoded_batch]


class MyDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super().__init__(root, transform)
		self.data, self.slices = torch.load(self.processed_paths[0])
		
	@property
	def raw_file_names(self):
		return ['data_list.pkl']  # 원시 파일 이름
	@property
	def processed_file_names(self):
		return ['data.pt']  # 처리된 파일 이름
		
	def process(self):
		with open(self.raw_paths[0], 'rb') as f:
			data_list = pickle.load(f)

		tokenizer = ByteLevelBPETokenizer('models/tokenizer/vocab.json', 'models/tokenizer/merges.txt')
		
		token_list = [data.x for data in data_list]
		encoded_attributes = []
		
		for tokens in token_list:
			temp = [tokenizer.encode(token) for token in tokens]
			encoded_attributes.append(temp)

		#encoded_attributes = tokenizer.encode_batch([data.x for data in data_list])
		node_indices = tokens_to_indices(encoded_attributes, tokenizer)
		node_lengths = [len(indices) for indices in node_indices]
		node_lengths = torch.tensor(node_lengths, dtype=torch.long)

		max_len = max(max(len(lst) for lst in graph) for graph in node_indices)
		padded_node_indices = [[lst + [0] * (max_len - len(lst)) for lst in graph] for graph in node_indices]
		padded_node_indices_tensor = [torch.tensor(graph, dtype=torch.long) for graph in padded_node_indices]
		
		temp = []
		for idx, data in enumerate(data_list):
			data.x = padded_node_indices_tensor[idx]
			temp.append(data)
		
		data, slices = self.collate(temp)
		torch.save((data, slices), self.processed_paths[0])
			


