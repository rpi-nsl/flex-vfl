import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver_VFL_Q import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size, labeled_frac=args.labeled_frac)
    #train_sub_config = get_config(dataset, mode='train_sub', batch_size=args.batch_size, labeled_frac=args.labeled_frac)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, labeled_frac=args.labeled_frac)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, labeled_frac=args.labeled_frac)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True)
    #train_loader_sub = get_loader(args, train_sub_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')
    
    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    solver = Solver(args, train_loader=train_loader, #train_loader_sub=train_loader_sub, 
                    dev_loader=valid_loader, test_loader=test_loader, is_train=True)
    solver.train_and_eval()
