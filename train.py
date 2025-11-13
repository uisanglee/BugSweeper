# General
import numpy as np
import random
import os
import json

import config as config
import model as mdl
import pool_model as pmdl
import utils
from utils import MyDataset

import optuna
from parser import parameter_parser

import argparse
import time

# Pytorch
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

from torch_geometric.data import DataLoader

import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


import GCL.augmentors as A

from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

from multiprocessing import Pool, cpu_count

import torch.multiprocessing as mp

from utils import MyDataset

# Global Variables

criterion = None
setting = None 
best_model = None
Graph = None

args = parameter_parser()

if args.gpu == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

setting = args.setting
level = args.level
batch_size = args.batch_size
embedding_dim = args.embedding_dim
num_layers = args.num_layers

print('Load Tokenizer')

tokenizer = ByteLevelBPETokenizer('models/tokenizer/vocab.json', 'models/tokenizer/merges.txt')

save_path = None
best_val_acc = 0
best_threshold = 0.5
all_losses = {}
eps = 10e-4

def create_sample_weights(embeddings, min_weight=0.1):
	simlarity_matrix = cosine_similarity(embeddings)

	np.fill_diagonal(simlarity_matrix, 0)  # Set diagonal to 0 to ignore self-similarity
	max_sim = np.max(simlarity_matrix, axis=1)

	weights = 1.0 - max_sim
	weights = np.clip(weights, min_weight, 1.0)

	return torch.tensor(weights, dtype=torch.float)

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

ranseed = 42
set_seed(ranseed)
print(f'seed:{ranseed}')

print('Using Device:', device)
print('Current cuda device:', torch.cuda.current_device())
if device.type == 'cuda': print(torch.cuda.get_device_name(0))
			
@torch.no_grad()

def compute_logit_adjustment(class_counts, tau=1.0):
	prior = class_counts.float() / class_counts.sum()
	adjustment = tau * torch.log(prior + 1e-12)
	return adjustment
		
def valid(loader, model, rank, epoch, logit_adjustment=None):
	"""
    Validate the model on a validation DataLoader.

    Args:
        loader (DataLoader): Validation data loader.
        model (torch.nn.Module): Model to evaluate.
        rank (torch.device): Device to run evaluation on.
        epoch (int): Current epoch number, used for logging.
        logit_adjustment (torch.Tensor, optional): Tensor to adjust logits for class imbalance.

    Returns:
        None: Prints validation metrics and saves best model by F1 score.
    """
	global best_threshold, best_val_acc, eps, criterion, setting, args

	model.eval()
	
	total_examples = 0
	val_preds = []
	val_labels = []
	
	name_list = []
	
	for i, batch in enumerate(loader):
		batch = batch.to(rank)
		if args.model == 'POOL' or args.model == 'rggnn' or args.model == 'rggnn_gat':
			output = model(batch.x, batch.edge_index.to(rank), batch.batch, batch.pool)
		else:
			output = model(batch.x, batch.edge_index.to(rank), batch.batch)
		
		if logit_adjustment is not None:
			output = output + logit_adjustment.unsqueeze(0)
			
		y = batch.y.to(rank)

		loss = criterion(output, y)
		total_examples += output.shape[0]
		
		preds = output.argmax(dim=1) # .tolist()
		
		y = y.cpu().tolist()
		val_preds.extend(preds.cpu().tolist())
		val_labels.extend(y)

	val_f1 = f1_score(val_labels, val_preds, average='macro')
	recall = recall_score(val_labels, val_preds, average='macro')
	precision = precision_score(val_labels, val_preds, average='macro')
		
	res = "\t".join(["EPOCH {} : ".format(epoch),"recall_val = {:.5f}".format(recall), "precision_val = {:.5f}".format(precision), "val_f1 = {:.5f}".format(val_f1)])
	print("\r",res,end="")
	
	if val_f1 > best_val_acc + eps:
		best_val_acc = val_f1
		
		with open(f'./models/{args.coverage}_{args.epochs}_{args.loss}.pth', 'wb') as f:
			try:
				torch.save(model.module.state_dict(), f)
			except:
				torch.save(model.state_dict(), f)
					
def contract_test(model, load_path=None, logit_adjustment=None):
	"""
    Evaluate a saved model on the test set at the contract level.
  
    Args:
        model (torch.nn.Module): Model architecture to load weights into.
        load_path (str): Path to the saved checkpoint.
        logit_adjustment (torch.Tensor, optional): Tensor to adjust logits.

    Returns:
        None: Prints detailed per-class metrics.
    """
	
	global test_data, best_model, best_val_acc, best_threshold, all_losses, eps, args, setting, batch_size, tokenizer
	print(f'best_threshold : {best_threshold}')
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	
	pred = 0
	
	model.load_state_dict(torch.load(load_path))
	model.to(device)
	model.eval()
	
	if best_model is not None:
		best_model.to(device)
		
	all_preds_list = []
	all_labels_list = []
	all_names = []
	
	contract_preds = []
	contract_labels = []
	
	result_dict = {}
	
	contract_name = None
	test_loader = DataLoader(test_data, batch_size=batch_size, sampler = SequentialSampler(test_data))
	
	class_count = torch.bincount(torch.tensor(test_data.y))
	
	logit_adjustment = compute_logit_adjustment(class_count, tau=1.0).to(device)

	with torch.no_grad():
		for data in tqdm(test_loader):
								
			data.to(device)
			data.edge_attr = data.edge_attr.type(torch.float32)
			if args.model == 'POOL' or args.model == 'rggnn' or args.model == 'rggnn_gat':
				output = model(data.x, data.edge_index, data.batch, data.pool)
			else:
				output = model(data.x, data.edge_index, data.batch)
			
			if logit_adjustment is not None:
				output = output + logit_adjustment.unsqueeze(0)
			
			preds = F.softmax(output, dim=1).argmax(dim=1)

			labels = data.y
			
			preds = preds.to('cpu')
			labels = labels.to('cpu')
			
			all_preds_list.extend(preds)
			all_labels_list.extend(labels)
			
			if setting == 'test':
				all_names.extend(data.name)

	all_preds = torch.tensor(all_preds_list) 
	all_labels = torch.tensor(all_labels_list)
		
	print('Macro Metrics ----------------------------------------------------------------------')
	recall = recall_score(all_labels, all_preds, average='macro')
	precision = precision_score(all_labels, all_preds, average='macro')
	f1 = f1_score(all_labels, all_preds, average='macro')
	print('Recall: {}, Precision : {}, F1 : {}'.format(recall, precision, f1))
	
	print(f'f1 0 : {f1_score(all_labels, all_preds, labels = [0], average = None)}')
	print(f'f1 1 : {f1_score(all_labels, all_preds, labels = [1], average = None)}')
	print(f'f1 2 : {f1_score(all_labels, all_preds, labels = [2], average = None)}')
	print(f'f1 3 : {f1_score(all_labels, all_preds, labels = [3], average = None)}')
	
	print('Precision 0 : {}, Recall 0 : {}'.format(precision_score(all_labels, all_preds, labels = [0], average = None), recall_score(all_labels, all_preds, labels = [0], average = None)))
	print('Precision 1 : {}, Recall 1 : {}'.format(precision_score(all_labels, all_preds, labels = [1], average = None), recall_score(all_labels, all_preds, labels = [1], average = None)))
	print('Precision 2 : {}, Recall 2 : {}'.format(precision_score(all_labels, all_preds, labels = [2], average = None), recall_score(all_labels, all_preds, labels = [2], average = None)))
	print('Precision 3 : {}, Recall 3 : {}'.format(precision_score(all_labels, all_preds, labels = [3], average = None), recall_score(all_labels, all_preds, labels = [3], average = None)))
	
	
	return 0
	
def run(train_data, val_data):
	"""
    Train the model over epochs, validate periodically, and finally test.

    Args:
        train_data (Dataset): Training dataset.
        val_data   (Dataset): Validation dataset.

    Returns:
        None
    """
	global args, tokenizer, criterion, best_val_acc, setting
	
	batch_size = args.batch_size
	class_count = torch.bincount(torch.tensor(train_data.y))

	tau = 1.0
	logit_adjustment = compute_logit_adjustment(class_count, tau).to(device)

	class_counts = class_count.float()
	sampler = RandomSampler(train_data)

	train_loader = DataLoader(train_data, batch_size=batch_size, sampler =sampler)
	val_loader = DataLoader(val_data, batch_size=batch_size, sampler = SequentialSampler(val_data))
	
	torch.manual_seed(42)
	
	if args.loss == 'ce':
		criterion = nn.CrossEntropyLoss()
		
	if args.model == 'POOL':
		vocab_size = len(tokenizer.get_vocab())+1
		model = pmdl.GNNModel(args.embedding_dim, args.hidden_channels,args.out_channels, 3, 3, vocab_size).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	
	if args.load != 'False':
		contract_test(model, args.load, logit_adjustment)
		quit()
	
	for epoch in range(args.epochs):
		model.train()

		for data in train_loader:
			torch.cuda.empty_cache()
			data = data.to(device)
			optimizer.zero_grad()
			if args.model == 'POOL' or args.model == 'rggnn' or args.model == 'rggnn_gat':
				output = model(data.x, data.edge_index, data.batch, data.pool)
			else:
				output = model(data.x, data.edge_index, data.batch)
			
			output = output + logit_adjustment.unsqueeze(0)
			loss = criterion(output, data.y)
			loss.backward()
			optimizer.step()
			
		val_acc = valid(val_loader, model, device, epoch, None)
		
	contract_test(model, f'./models/{args.coverage}_{args.epochs}_{args.loss}.pth', None)

def main():
	global setting, Graph, setting_type, best_model, best_val_acc, best_threshold, all_losses, eps, args, criterion, batch_size, tokenizer, test_data

	coverage = str(args.coverage)
		
	train_data = MyDataset(root = config.DATASET_DIR / level / 'train' / coverage)
	val_data = MyDataset(root = config.DATASET_DIR  / level /  'valid' / coverage)
	curr_losses = []
	
	if setting == 'train':
		test_data = MyDataset(root = config.DATASET_DIR / level / 'test' / coverage)
		run(train_data, val_data)
		
	elif setting == 'test':
		test_data = MyDataset(root = config.DATASET_DIR / level / 'test' / coverage)
		run(train_data, val_data)
			
	
if __name__ == '__main__':
	main()