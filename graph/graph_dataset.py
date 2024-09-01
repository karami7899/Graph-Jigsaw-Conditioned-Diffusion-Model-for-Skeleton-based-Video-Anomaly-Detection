import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from graph.util.env import get_device, set_device
from graph.util.preprocess import build_loc_net, construct_data
from graph.util.net_struct import get_feature_map, get_fc_graph_struc
from graph.util.iostream import printsep

from graph.datasets.TimeDataset import TimeDataset


from graph.models.GDN import GDN



import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

from typing import Dict, List, Tuple


def make_graph(input_data:List[torch.Tensor]):
  num_samples, dimensions, num_frame, features = input_data.shape
  all_features = dimensions * features
  
  new_tensor = input_data.permute(0, 2, 1, 3).contiguous() #
  reshaped_tensor_3d = new_tensor.reshape(num_samples, num_frame, all_features).contiguous() #
  reshaped_tensor_2d = new_tensor.reshape(num_samples*num_frame, all_features).contiguous() #

  feature_map = get_feature_map(all_features)
  fc_struc = get_fc_graph_struc(all_features)

  fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
  fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

  cfg = {
    'slide_win': 3, 
    'slide_stride': 4, 
  }
  train_dataset_indata = construct_data(reshaped_tensor_2d.permute(1,0).contiguous(), feature_map, labels=0)
  train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
  
  train_dataloader = DataLoader(train_dataset, batch_size=1024,
                            shuffle=False, num_workers=0)
  
  return train_dataloader
 
  