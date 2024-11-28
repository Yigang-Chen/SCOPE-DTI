# ============ Reading configs ==============

from configs import get_cfg_defaults
import os
import sys
from time import time
from utils import set_seed, graph_collate_func, mkdir, print_with_time

s = time()   # Start time

cfg = get_cfg_defaults()
if len(sys.argv) < 2:
    print_with_time("Please provide a config file")
    sys.exit(1)
cfg_path = sys.argv[1]
if not os.path.exists(cfg_path):
    print_with_time(f"Config file {cfg_path} does not exist")
    sys.exit(1)
cfg.merge_from_file(cfg_path)
cfg.merge_from_list(sys.argv[2:])
cfg.freeze()
mkdir(cfg.RESULT.OUTPUT_DIR)
# experiment = None
print_with_time(f"Config yaml: {cfg_path}")
print_with_time(f"Hyperparameters:")
print_with_time(dict(cfg))

# ========== Importing all modules ===========

from models import Scope
from trainer import Trainer
import torch
import warnings
import pandas as pd
import numpy as np
from dataloader import DTIDataset, ConditionalInitiableDataset
from torch.utils.data import DataLoader
import pickle as pkl

print_with_time("Imported all modules")

# =================== Main ===================

set_seed(cfg.SOLVER.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print_with_time(f"Running on: {device}")
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

df_train = pd.read_parquet(cfg.DATA.TRAIN)
df_val = pd.read_parquet(cfg.DATA.VAL)
df_test = pd.read_parquet(cfg.DATA.TEST)
print_with_time("Loaded data")

dataset = ConditionalInitiableDataset(**cfg)

train_dataset = DTIDataset(df_train.index.values, df_train, father=dataset)
val_dataset = DTIDataset(df_val.index.values, df_val, father=dataset)
test_dataset = DTIDataset(df_test.index.values, df_test, father=dataset)

params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 
          'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 
          'collate_fn': graph_collate_func, 'pin_memory': True}
training_generator = DataLoader(train_dataset, **params)
params['shuffle'] = False
params['drop_last'] = False
val_generator = DataLoader(val_dataset, **params)
test_generator = DataLoader(test_dataset, **params)

model = Scope(**cfg).to(device)
with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
    wf.write(str(model))
print_with_time(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True

trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)
result = trainer.train()

e = time()  # End time

print(f"Total running time: {round(e - s, 2)}s")  # Print the total running time