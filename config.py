import re
import os
import sys

from pathlib import Path

###################################################### Directories ######################################################

PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / 'preprocessed' 
CODE_PATH = PROJECT_ROOT / 'datasets'
AST_PATH = PROJECT_ROOT / 'solc_ast'

################################# Regular Expressions for parsing solidity files ########################################

filenum = re.compile('([buggy]*[safe]*)[_]([0-9]*)')
vp = re.compile('(pragma)\s(solidity)([^a-zA-Z0-9]+)([0-9]+[.][0-9]+[.][0-9]+)')
cmdv = re.compile('[0-9]+[.][0-9]+[.][0-9]+')

dump_info = re.compile("JSON|=======")
src_info = re.compile("([0-9]*)[:](.*)")

func_decl = re.compile(r'(function\s+([a-zA-Z$_][a-zA-Z0-9$_]*)?\s*\(|function\s*\(\s*\)|constructor\s*\()', re.DOTALL)

elementary_type_name = r'address\spayable|address|bool|string|bytes[0-9]*|int[0-9]*|uint[0-9]*|fixed|ufixed|event'
identifier = r'[a-zA-Z$_]+[a-zA-Z0-9$_]*'
modifiers = r'public|private|internal|constant|immutable|memory|storage|calldata'

var_declaration = rf'({elementary_type_name})\s+(({modifiers})\s+)?'
mapping_declaration = rf'mapping\s*\(\s*.*\s*\)\s+(({modifiers})\s+)?'
variable_declaration = rf'({var_declaration}|{mapping_declaration}){identifier}(\s*.*)*'

var_decl = re.compile(variable_declaration)
contractdecl = re.compile(r'(contract|library)\s+(\w+)')
modifier_decl = re.compile(r'modifier\s+([a-zA-Z$_][a-zA-Z0-9$_]*)\s*\(')

############################################# preprocess types ##########################################################

pass_list = ['indexed','fullyImplemented','contractKind','exportedSymbols','names','lValueRequested','member_name','value','token','hexvalue','name','isConstant','isLValue','isPure','subdenomination','absolutePath','linearizedBaseContracts','contractDependencies','argumentTypes','overloadedDeclarations', 'stateMutability', 'stateVariable', 'storageLocation', 'commonType', ]
SOLC_VERSIONS = ["0.8.9", "0.8.8", "0.8.7", "0.8.6", "0.8.5", "0.8.4", "0.8.3", "0.8.20", "0.8.2", "0.8.19", "0.8.18", "0.8.17", "0.8.16", "0.8.15", "0.8.14", "0.8.13", "0.8.12", "0.8.11", "0.8.10", "0.8.1", "0.8.0", "0.7.6", "0.7.5", "0.7.4", "0.7.3", "0.7.2", "0.7.1", "0.7.0", "0.6.9", "0.6.8", "0.6.7", "0.6.6", "0.6.5", "0.6.4", "0.6.3", "0.6.2", "0.6.12", "0.6.11", "0.6.10", "0.6.1", "0.6.0", "0.5.9", "0.5.8", "0.5.7", "0.5.6", "0.5.5", "0.5.4", "0.5.3", "0.5.2", "0.5.17", "0.5.16", "0.5.15", "0.5.14", "0.5.13", "0.5.12", "0.5.11", "0.5.10", "0.5.1", "0.5.0", "0.4.26", "0.4.25", "0.4.24", "0.4.23", "0.4.22", "0.4.21", "0.4.20"]

CLASS_LIST = ['reentrancy', 'unchecked_low_calls', 'time_manipulation']

class_dict = {'other':0, 'reentrancy':1, 'unchecked_low_calls':2, 'time_manipulation':3}
DUMP_LIST = ['literals', 'contractKind', 'fullyImplemented', 'constant', 'documentation', 'implemented', 'isConstructor', 'kind', 'stateVariable', 'storageLocation', 'isLValue', 'isPure', 'hexvalue', 'isStructConstructorCall', 'lValueRequested', 'type_conversion', 'isConstant','name', 'ContractDependencies','linearizedBaseContracts','argumentTypes','overloadedDeclarations']
