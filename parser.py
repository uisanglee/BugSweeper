import argparse


def parameter_parser():
	# Experiment parameters
	parser = argparse.ArgumentParser(description='Smart Contracts Vulnerability Detection')
 
	parser.add_argument('-S', '--setting', type=str, default='train')
	parser.add_argument('-M', '--model', type=str, default='POOL', choices=['POOL'])
	parser.add_argument('-L', '--level', type=str, default='function', choices=['function', 'reentrancy'])

	parser.add_argument('--lr', type=float, default=1e-4, help='VEthVul learning rate') 
	parser.add_argument('--wd', type=float, default=1e-5, help='VEthVul weight decay')

	parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
	parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')

	parser.add_argument('--epochs', type=int, default=500, help='number of epochs')

	parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimension')
	parser.add_argument('--hidden_channels', type=int, default=1024, help='hidden dimension') 
	parser.add_argument('--out_channels', type=int, default=4, help='output dimension')

	parser.add_argument('--loss', type=str, default='ce', choices=['ce'])
	
	parser.add_argument('--input', type=str, default='function', help='input type', choices=['function'])
	
	parser.add_argument('--coverage', type=str, default=4, help='coverage')
	parser.add_argument('--load', type=str, default='False', help='load model')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--pool_mode', type=str, default='hier', choices=['single', 'hier'],
		help='single: use only base pool assignment, hier: use all available hierarchical levels')
	parser.add_argument('--max_pool_levels', type=int, default=0,
		help='0 means use all available levels; positive value caps hierarchy depth used by model')
	
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')

	return parser.parse_args()
