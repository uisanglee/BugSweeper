import os, sys, json, re, time
import subprocess
import chardet
import argparse
import torch
import random
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import config as config

from networkx.readwrite import json_graph

from pathlib import Path

from utils import MyDataset

from torch_geometric.utils import from_networkx
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#GLOBAL VARIABLES

DUMP_LIST = config.DUMP_LIST
dump_info = config.dump_info
pass_list = config.pass_list
class_dict = config.class_dict

solc_version_list = config.SOLC_VERSIONS
Graph = None

DATA_LIST = []

prev_size = 0
setting = None
level = 'function'

#REGEX
vp = config.vp
cmdv = config.cmdv
filenum = config.filenum
src_info = config.src_info
func_decl = config.func_decl
var_decl = config.var_decl
modifier_decl = config.modifier_decl

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

set_seed(42)


def read_graphs(subgraph, name=None):
	"""
    Convert a NetworkX subgraph into a PyTorch Geometric graph object.
    - Relabel nodes to [0..N-1]
    - Build edge_index and edge_attr tensors
    - Copy graph-level label if present
    """
	
	mapping = dict(zip(subgraph, range(0, subgraph.number_of_nodes())))
	subgraph = nx.relabel_nodes(subgraph, mapping)
	edge_index = torch.tensor(list(subgraph.edges)).t().contiguous()

	edge_attr = np.zeros([edge_index.shape[1], 12])
	i = 0
	
	for u, v, attr in subgraph.edges(data=True):
		if attr['edge_attr'] == 'Child':
			edge_attr[i][0] = 1
		
		elif attr['edge_attr'] == 'ReferencedDeclaration':
			edge_attr[i][1] = 1
			
		elif attr['edge_attr'] == 'FunctionReturnParameter':
			edge_attr[i][2] = 1
		
		elif attr['edge_attr'] == 'SuperFunction':
			edge_attr[i][3] = 1
		
		elif attr['edge_attr'] == 'Assignment':
			edge_attr[i][4] = 1
			
		elif attr['edge_attr'] == 'CondTrue':
			edge_attr[i][5] = 1
		
		elif attr['edge_attr'] == 'CondFalse':
			edge_attr[i][6] = 1
		
		elif attr['edge_attr'] == 'WhileExecution':
			edge_attr[i][7] = 1
					
		elif attr['edge_attr'] == 'WhileNext':
			edge_attr[i][8] = 1
			
		elif attr['edge_attr'] == 'ForExecution':
			edge_attr[i][9] = 1
						
		elif attr['edge_attr'] == 'ForNext':
			edge_attr[i][10] = 1
			
		elif attr['edge_attr'] == 'NextStatement':
			edge_attr[i][11] = 1
		
		i = i+1
			
	for idx, attr in subgraph.nodes(data=True):
		x = attr['node_type']
		subgraph.nodes[idx]['x'] = x 
		key_list = list(attr.keys())

		for key in key_list:
			if key != 'x':
				del subgraph.nodes[idx][key]

	edge_attr = torch.tensor(edge_attr)
	new_G = from_networkx(subgraph)
	
	new_G.edge_attr = edge_attr
	
	if subgraph.graph != {}:
		
		new_G.y = subgraph.graph['graph_type']
		
		del new_G.graph_type
	
	new_G.edge_index = edge_index
	
	if name != None:
		new_G.name = name
			
	return new_G

def merge_subgraph(all_subgraphs, subgraphs, coverage, Flag = False):
	"""
    Recursively merge function-level subgraphs to include referenced functions/variables up to a given depth.
    If Flag=True, only include FunctionDefinition and ModifierDefinition nodes.
    """
	
	global Graph
	required_subgraphs = {}
	
	# 끝나는 조건
	if coverage < 0:
		return subgraphs

	if Flag == True:
		for subgraph_key in subgraphs:
			subgraph = subgraphs[subgraph_key]
			# 원래는 Function Definition 만
			if subgraph.nodes[subgraph_key]['node_type'] in ['FunctionDefinition','ModifierDefinition']:
				required_subgraphs[subgraph_key] = subgraph
		
	
	else:
		for subgraph_key in subgraphs:
			subgraph = subgraphs[subgraph_key]

			merged = subgraph.copy()
	
			for idx in subgraph.nodes():
				node = subgraph.nodes[idx]

				for attr in ['referencedDeclaration', 'functionReturnParameters','superFunction','assignments']:

					# assignments는 list로 구성되어 있음
					if attr == 'assignments':
						if attr in node.keys():
							temp = node[attr]

							for j in temp:
								if j in all_subgraphs.keys():
									temp_2 = {}
									temp_2[j] = all_subgraphs[j]
									next_ = merge_subgraph(all_subgraphs, temp_2, coverage-1)

									for dst in next_:
										merged = nx.compose(merged, next_[dst])
										merged.add_edge(idx, dst, edge_attr = 'Assignment')
		
					else: 
						if attr in node.keys() and node[attr] != None:
							temp_2 = {}

							if node[attr] in all_subgraphs.keys():
								temp_2[node[attr]] = all_subgraphs[node[attr]]
								next_ = merge_subgraph(all_subgraphs, temp_2, coverage-1)

								for dst in next_:
									merged = nx.compose(merged, next_[dst])
									merged.add_edge(idx, dst, edge_attr = attr[0].upper() + attr[1:])

							elif Graph.nodes[node[attr]]['node_type'] == 'Sender':
								temp_2[node[attr]] = Graph.subgraph(node[attr]).copy()
								next_ = merge_subgraph(all_subgraphs, temp_2, coverage-1)

								for dst in next_:
									merged = nx.compose(merged, next_[dst])
									merged.add_edge(idx, dst, edge_attr = attr[0].upper() + attr[1:])

			required_subgraphs[subgraph_key] = merged
		return required_subgraphs
	
	return merge_subgraph(all_subgraphs, required_subgraphs, coverage-1)

def sampling(G, class_name, src_dict = None):
	"""
    Extract function-level subgraphs (FLAGs) from the full contract graph G,
    annotate each with its vulnerability label if src_dict is provided.
    Returns dict of sampled subgraphs and pool indices.
    """
	global coverage, Graph, class_dict
	Graph = G
	Found = False
	subgraph_dict = {}
	
	# Each Contract Level
	for i in G.successors(0):	
		if G.nodes[i]['node_type'] == 'ContractDefinition':
			Found = True
			_contract = G.nodes[i]
			contract_idx = i

			# Each Function Level
			for j in G.successors(contract_idx):
				subgraph_dict[j] = nx.dfs_tree(G, source=j)
	
	if Found == False:
		if G.nodes[0]['node_type'] == 'ContractDefinition':
			Found = True
			_contract = G.nodes[0]
			
			for j in G.successors(0):
				subgraph_dict[j] = nx.dfs_tree(G, source=j)   
	
	#Edge construction
	for idx, n in G.nodes(data=True):
		keylist = [i for i in G.nodes[idx].keys()]
		templist = ['referencedDeclaration', 'functionReturnParameters','superFunction','assignments']
		intersec = list(set(keylist) & set(templist))
		
		for i in intersec:
			dst = G.nodes[idx][i]

			if dst == "None":
				continue

			if i == 'referencedDeclaration':
				G.add_edge(idx, int(dst), edge_attr = 'ReferencedDeclaration') #2
				
			elif i == 'functionReturnParameters':
				G.add_edge(idx, int(dst), edge_attr = 'FunctionReturnParameter') #3

			elif i == 'superFunction':
				G.add_edge(idx, int(dst), edge_attr = 'SuperFunction') # 4

			elif i == 'assignments':
				for j in dst:
					G.add_edge(idx, int(j), edge_attr = 'Assignment') # 5
				
		
		if G.nodes[idx]['node_type'] == 'IfStatement':
			temp = G.successors(idx)
			cond = None   
			true = None
			false = None
			
			for k in temp:
				if cond == None:
					cond = k
				elif true == None:
					true = k
				elif false == None:
					false = k 
			
			G.add_edge(cond, true, edge_attr = 'CondTrue') #6
			
			if false != None:
				G.add_edge(cond, false, edge_attr = 'CondFalse')#7
				
		if G.nodes[idx]['node_type'] == 'WhileStatement':
			temp = G.successors(idx)
			cond = None   
			true = None
			
			for k in temp:
				if cond == None:
					cond = k
				elif true == None:
					true = k
			
			G.add_edge(cond, true, edge_attr = 'WhileExecution') #8, 10
			G.add_edge(true, cond, edge_attr = 'WhileNext')#, 8, 11
		
		if G.nodes[idx]['node_type'] == 'ForStatement':
			temp = G.successors(idx)
			cond = None   
			true = None
			
			for k in temp:
				if cond == None:
					cond = k
				elif true == None:
					true = k
			
			G.add_edge(cond, true, edge_attr = 'ForExecution') #9, 10
			G.add_edge(true, cond, edge_attr = 'ForNext') #9, 11
		
		if G.nodes[idx]['node_type'] == 'Block':
			temp = G.successors(idx)
			prev = None
			for k in temp:
				if prev != None:
					G.add_edge(prev, k, edge_attr = 'NextStatement') # 12
					
				prev = k
				
	#Subgraph construction
	subgraphs = {}
	pool_idx = []
	
	for i in subgraph_dict:
		nodes = list(subgraph_dict[i].nodes)
		subgraphs[i] = G.subgraph(nodes).copy()
		pool_idx.append(nodes)		
		
	sampled = merge_subgraph(subgraphs, subgraphs, coverage, True)

	#src dict 에 포함된 경우에만 추출
	if src_dict != {}:

		vul_samples = {}
		srcidx_end = list(src_dict.keys())[-1]
		srcidx = 1

		src_start = src_dict[srcidx][1]
		src_end = src_dict[srcidx][2]
		sampled_key = list(sampled.keys())

		for i in sampled_key:

			sampled[i].graph = {"graph_type": torch.tensor([src_dict[srcidx][0]]).type(torch.long)}
			src = int(src_info.search(G.nodes[i].pop('src')).group(1))
			while src >= src_end and srcidx < srcidx_end:
				srcidx = srcidx + 1
				src_start = src_dict[srcidx][1]
				src_end = src_dict[srcidx][2]

			if src >= src_start and src <= src_end:
				vul_samples[i] = sampled.pop(i)
				
				vul_samples[i].graph =	{"graph_type": torch.tensor([src_dict[srcidx][0]]).type(torch.long)}

	#만약 Test 가 True 일 때, Test set 에서는 모든 Subgraph 를 추출		
	else:
		vul_samples = sampled
		
	return vul_samples, pool_idx

def preprocess_solc_json(name, enc, class_name, save_file):
	"""
    Load solc AST JSON, filter out unwanted lines, parse into a NetworkX DiGraph,
    and normalize node attributes for graph construction.
    Returns cleaned graph_.
    """
	filename = save_file

	try:
		with open(filename, 'r', encoding = enc) as f:
			lines = f.readlines()

	except:
		enc = 'utf-8'
		with open(filename, 'r', encoding=enc) as f:
			lines = f.readlines()

	with open(filename, 'w', encoding=enc) as f:
		for line in lines:
			if re.match(dump_info, line) == None:
				f.write(line)

	with open(filename, 'r', encoding=enc) as f:
		ast = json.load(f)

	try:
		ast_graph = nx.DiGraph(json_graph.tree_graph(ast))
	except:
		ast_graph = nx.DiGraph(json_graph.tree_graph(ast['children'][0]))

		if ast_graph.number_of_nodes() == 1:
			ast_graph = nx.DiGraph(json_graph.tree_graph(ast['children'][1]))

	graph_ = nx.DiGraph.copy(ast_graph)
	nodenum = ast_graph.number_of_nodes() + 1000
	dfs_idx = 0

	nx.set_edge_attributes(graph_, 'Child', 'edge_attr')

	for idx in ast_graph.nodes():
		dfs_idx = dfs_idx + 1
		graph_.nodes[idx]['index'] = dfs_idx

		if 'attributes' in list(graph_.nodes[idx].keys()):
			attributes = graph_.nodes[idx]["attributes"]
			for i in attributes:
				if i in pass_list:
					continue
				elif isinstance(attributes[i], dict) or isinstance(attributes[i], bool):
					continue

				if attributes[i] is not None:
					if isinstance(attributes[i], int):
						graph_.nodes[idx][i] = attributes[i]

					elif isinstance(attributes[i], list):
						if attributes[i][0] == None:
							continue

						elif i == "assignments":
							graph_.nodes[idx][i] = attributes[i]

						elif isinstance(attributes[i][0], dict):
							continue

						else : continue
					
					#also for Inline assembly
					elif i == "operations":
						continue

					elif i == 'type':
						continue

					elif attributes[i] == "":
						continue
					
					elif i == 'visibility':
						if attributes[i] == 'private' or attributes[i] == 'external':
							graph_.nodes[idx][i] = attributes[i][0].upper() + attributes[i][1:]

					else:
						graph_.nodes[idx][i] = str(attributes[i])[0].upper() + str(attributes[i])[1:]

			del graph_.nodes[idx]["attributes"]
			
		graph_.nodes[idx]['node_type'] = graph_.nodes[idx].pop('name')
		
		keylist = [i for i in graph_.nodes[idx].keys()]
		intersec = list(set(keylist) & set(DUMP_LIST))

		for dump in intersec:
			del graph_.nodes[idx][dump]

		keylist = [i for i in graph_.nodes[idx].keys()]
		remove_list = ['node_type', 'src', 'index', 'functionReturnParameters','superFunction','assignments','scope']
		intersec = list(set(keylist) & set(remove_list))

		for i in intersec:
			keylist.remove(i)

		for i in keylist:
			if i == "referencedDeclaration" and graph_.nodes[idx][i] != 'None':
				sender = int(graph_.nodes[idx][i])
				if sender not in graph_.nodes:
					dfs_idx = dfs_idx + 1
					graph_.add_node(sender, index = dfs_idx, node_type = "Sender")
					graph_.add_edge(idx, sender, edge_attr='ReferencedDeclaration') #referencedDeclaration
					del graph_.nodes[idx][i]

				continue
				
			try:
				if(graph_.nodes[idx][i] == "None" or graph_.nodes[idx][i] == "[None]"):
					del graph_.nodes[idx][i]
					continue
			except:
				pass
			
			nodenum = nodenum + 1
			dfs_idx = dfs_idx + 1

			temp = graph_.nodes[idx].pop(i)
			i = i[0].upper() + i[1:]
			
			graph_.add_node(nodenum, node_type = i, index = dfs_idx)
			graph_.add_edge(idx, nodenum, edge_attr='Child') #child

			nodenum = nodenum + 1
			dfs_idx = dfs_idx + 1
			
			if isinstance(temp, str):
				temp = temp[0].upper()+temp[1:]

			graph_.add_node(nodenum, node_type = temp, index = dfs_idx)# Number가 더해진
			graph_.add_edge(nodenum-1, nodenum, edge_attr='Child') #child
			
	return graph_

def version_try(_code_v, file_path, save_file):
	"""
    Try compiling with decreasing solc versions until success or reaching minimum.
    Uses solc-select to switch compiler and checks JSON size.
    """
	idx = solc_version_list.index(_code_v) 
	code_v = solc_version_list[idx-1]

	ver_cmd = f"solc-select use {code_v} 1> /dev/null"
	os.system(ver_cmd)

	json_cmd = f"solc --ast-json {file_path} > {save_file} 2> /dev/null"
	os.system(json_cmd)

	size = int(os.path.getsize(save_file))

	if size < 100:
		del_cmd = f"rm -rf {save_file}"
		os.system(del_cmd)


		if(code_v == "0.8.0"):
			return 0

		else:
			temp = version_try(code_v, file_path, save_file)
			if(temp == 0):
				return 0

	return 1

def read_code(lines, DONE, buglog, file_path, save_file, solc_v):
	"""
    Scan source lines to map byte offsets to vulnerability labels based on buglog DataFrame.
    Also ensure correct solc version and regenerate AST JSON if needed.
    Returns detected solc version and src_dict mapping.
    """
	src_dict = {}
	src_byte = 0
	code_v = "undef"

	loc_start = 999999999
	loc_end = 0
	bug_idx = 0

	fp = open(file_path, 'rb')
	src_lines = fp.readlines()
	label = 0

	for i, line in enumerate(src_lines):

		if buglog is not None:
			
			spaces = len(line) - len(line.lstrip())
			temp_byte = src_byte + spaces
			loc = i+1
			
			temp = buglog.query(f'loc == {loc}')

			if len(temp) == 1:
				temp = temp.to_dict('list')
				
				if temp['bug type'][0] == 'Safe':
					label = 0
					
				elif temp['bug type'][0] == 'reentrancy' or temp['bug type'][0] == 'Reentrancy':
					label = 1
				
				elif temp['bug type'][0] == 'unchecked_low_calls' or temp['bug type'][0] == 'Unchecked_low_calls':
					label = 2
				
				elif temp['bug type'][0] == 'time_manipulation' or temp['bug type'][0] == 'Time_manipulation':
					label = 3

				bug_idx = bug_idx + 1
				loc_start = loc

				loc_end = loc_start + temp['length'][0]
				src_list = [label,temp_byte, 0]
				
			if loc <= loc_end:
				temp_line = line.decode('utf-8')
				src_list[-1] = temp_byte
				src_dict[bug_idx] = src_list
				
				if var_decl.search(temp_line) != None or func_decl.search(temp_line) != None:
					src_list[-1] = temp_byte
					src_dict[bug_idx] = src_list

				elif modifier_decl.search(temp_line) != None:
					src_list[-1] = temp_byte
					src_dict[bug_idx] = src_list

			src_byte = src_byte + len(line)

	for i, line in enumerate(lines):

		if DONE == True:
			code_v = "0.4.20"
			continue

		if vp.search(line) != None:
			keyword = vp.search(line).group(3)
			code_v = vp.search(line).group(4)

			if "^" in keyword or ">=" in keyword:
				flag = version_try("0.4.20", file_path, save_file)
				if flag == 0:
					del_cmd = "rm -rf {save_file}"
					os.system(del_cmd)

			else:
				if code_v != solc_v:
					ver_cmd = f"solc-select use {code_v} 1> /dev/null"
					os.system(ver_cmd)	
												
					json_cmd = f"solc --ast-json {file_path} > {save_file} 2> /dev/null"
					os.system(json_cmd)
							
					size = int(os.path.getsize(save_file))

					if size < 10:
						del_cmd = f"rm -rf {save_file}"
						os.system(del_cmd)

				else :
					json_cmd = f"solc --ast-json {file_path} > {save_file} 2> /dev/null"
					os.system(json_cmd)

					size = int(os.path.getsize(save_file))

					if size < 10:
						del_cmd = f"rm -rf {save_file}"
						os.system(del_cmd)

	return code_v, src_dict

def preproc(save_file, file_path, name, bug_log = None, error_file = None, class_name = None, cond = None):
	"""
    Main preprocessing pipeline for a single Solidity file:
    - Detect encoding
    - Run version-specific solc to produce AST JSON
    - Build networkx AST
    - Depending on level (contract/function), extract PyG graphs
    - Append to DATA_LIST
    """
	global DATA_LIST, setting, level

	subgraph = ""
	new_idx = 0

	data = str(subprocess.check_output(['solc', '--version']))
	solc_v = cmdv.search(data).group()

	rawdata = open(file_path, 'rb').read()
	result = chardet.detect(rawdata)
	enc = result['encoding']
	DONE = False

	if os.path.isfile(save_file):
		DONE = True

	with open(file_path, 'r+') as f:
		lines = f.readlines()
		code_v, src_dict = read_code(lines, DONE, bug_log, file_path, save_file, solc_v)

		if code_v == "undef":
			lines.insert(0, 'pragma solidity ^0.5.0;\n')
			f.seek(0)
			f.writelines(lines)
			flag = version_try("0.5.0", file_path, save_file)

			if flag == 0:
				print(f"\n{file_path} is not supported by solc")
				return 0
		
	source = preprocess_solc_json(name, enc, class_name, save_file)
	H = nx.DiGraph()

	H.add_nodes_from(sorted(source.nodes(data=True), key = lambda x:x[1]['index']))
	H.add_edges_from(source.edges(data=True))

	keys = []
	values = []

	for i, attr in H.nodes(data=True):
		keys.append(i)
		values.append(new_idx)
		new_idx = new_idx + 1

	dic = dict(zip(keys, values))

	for idx, attr in H.nodes(data=True):
		if 'referencedDeclaration' in list(attr.keys()):
			temp = int(attr['referencedDeclaration'])
			attr['referencedDeclaration'] = dic[temp]

		elif 'functionReturnParameters' in list(attr.keys()):
			temp = int(attr['functionReturnParameters'])
			attr['functionReturnParameters'] = dic[temp]

		elif 'superFunction' in list(attr.keys()):
			temp = int(attr['superFunction'])
			attr['superFunction'] = dic[temp]

		elif 'assignments' in list(attr.keys()):
			asgnlist = []
			for i in attr['assignments']:
				if i == None:
					continue
				
				temp = dic[int(i)]
				asgnlist.append(temp)
			
			attr['assignments'] = asgnlist

	H = nx.relabel_nodes(H, dic)
	subgraph, pool_idx = sampling(H, class_name, src_dict)
	
	idx2pool = {}
	
	for pool_id, idx_list in enumerate(pool_idx):
		for idx in idx_list:
			idx2pool[idx] = pool_id
	
	for idx in subgraph:
		sub_pool_ids = [ idx2pool[n] for n in subgraph[idx].nodes()]		
		
		pyg = read_graphs(subgraph[idx], name)
		pyg.pool = sub_pool_ids

		DATA_LIST.append(pyg)

	return DATA_LIST

def main():
	"""
    load or build dataset for train/valid/test or DApp modes,
    caching intermediate results to avoid reprocessing.
    """
	
	global coverage, prev_size, setting, DATA_LIST, level
	parser = argparse.ArgumentParser(description='Preprocess solidity files into Graphs')

	parser.add_argument('-c', '--coverage', type=int, default=4, help='Coverage percentage for training data')
	parser.add_argument('-m', '--mode', type=str, default='train', help='Mode of operation: train | test | valid | DApp ')
	parser.add_argument('-l', '--level', type=str, default='function', help='Level of operation: function | reentrany | DApp')
	
	args = parser.parse_args()

	coverage = args.coverage
	coverage_ = str(coverage)
	setting = args.mode
	level = args.level

	print('source codes preprocess.......')

	if setting in ['train', 'valid', 'test'] and level == 'function':

		if os.path.isfile(config.DATASET_DIR / level / setting / coverage_ / 'raw' / 'data_list.pkl'):
			with open(config.DATASET_DIR / level / setting / coverage_ / 'raw' /'data_list.pkl', 'rb') as f:
				DATA_LIST = pickle.load(f)

		if os.path.isfile(config.DATASET_DIR / level / setting / coverage_ / 'processed' / 'data.pt'):
			dataset = MyDataset(root = config.DATASET_DIR / level / setting / coverage_)
			return

	if not os.path.isfile(config.DATASET_DIR / level / setting / coverage_ / 'raw' / 'data_list.pkl'):
		
		Class_list = config.CLASS_LIST
		
		if setting == 'DApp':
			Class_list = ['reentrancy']
			
		for class_name in Class_list:
			preprocess_path = config.CODE_PATH / class_name
			error_file = []

			for (root, _, files) in os.walk(preprocess_path / setting):
				sol_files = [f for f in files if f.endswith('.sol')]
				for cur, file in enumerate(sol_files):
					total_count = len(sol_files)

					file_path = os.path.join(root, file)
					name, ext = os.path.splitext(file)

					print(f"\rProcessing {setting} | {class_name} | \t%.2f%% | {file}"%(float(cur)/total_count*100), end='')
					buglognum = name

					buglog = pd.read_csv(preprocess_path / setting / f'BugLog_{buglognum}.csv')
					buglog = buglog.sort_values(by='loc')

					save_file = config.AST_PATH / class_name / setting / f'{name}.json'
					
					preproc(save_file, file_path, name, buglog, error_file, class_name, setting)

					error_list = config.DATASET_DIR / 'error_list.txt'
					error_log = config.DATASET_DIR / 'error_log.txt'

					with open(error_list, 'w+') as file:
						file.write('\n'.join(error_file))

					if prev_size == len(DATA_LIST):
						with open(error_log, 'w+') as f:
							f.write(f'csv or None function declaration in {file_path}\n')
					
					prev_size = len(DATA_LIST)
			print(f"\n{class_name} is done")
	
	if setting in ['train','valid','test'] and level == 'function':
				
		with open(config.DATASET_DIR / level / setting / coverage_ / 'raw' /'data_list.pkl', 'wb') as f:
			pickle.dump(DATA_LIST, f)
			
		if setting == 'train':
			if not os.path.isfile('./models/tokenizer/vocab.json'):
				import itertools
				from tokenizers import ByteLevelBPETokenizer
				tokenizer = ByteLevelBPETokenizer()

				text_list = [data.x for data in DATA_LIST]
				text_list = list(itertools.chain(*text_list))

				tokenizer.train_from_iterator(text_list, vocab_size=50265, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
				tokenizer.save_model('./')
				return 0

		dataset = MyDataset(root = config.DATASET_DIR / level / setting / coverage_)

	elif setting in ['train','valid','test','pretrain'] and level == 'contract':
		if not os.path.isfile(config.CONTRACT_LEVEL_DIR / setting /'processed' / 'data.pt'):
			dataset = MyDataset(root = config.CONTRACT_LEVEL_DIR / setting)
			
	else :
		with open(config.DATASET_DIR / level / setting / coverage_ / 'raw' /'data_list.pkl', 'wb') as f:
			pickle.dump(DATA_LIST, f)
		
		dataset = MyDataset(root = config.DATASET_DIR / level / setting / coverage_)


if __name__ == "__main__":
	main()
