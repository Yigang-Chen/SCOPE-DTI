from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.ATOM_IN_DIM = [74, 1]
_C.DRUG.ATOM_HIDDEN_DIM = [320, 64]
_C.DRUG.EDGE_IN_DIM = [16, 1]
_C.DRUG.EDGE_HIDDEN_DIM = [32, 1]
_C.DRUG.NUM_LAYERS = 3
_C.DRUG.DROP_RATE = 0.1
_C.DRUG.MAX_NODES = 300
_C.DRUG.EDGE_CUTOFF = 4.5 # edge cutoff, unit is A
_C.DRUG.NUM_RDF = 16

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.EMBEDDING_DIM = 320
_C.PROTEIN.PADDING = True
_C.PROTEIN.MAX_LENGTH = 2000

_C.PROTEIN.GRAPH = CN()
_C.PROTEIN.GRAPH.COORD_PATH = "/share/home/grp-huangxd/chenyigang/data/pped_alphafold/unip_cords.pkl" # path of protein coordinates file
_C.PROTEIN.GRAPH.PATH = "protein_graph.pkl" # path to save protein graph
_C.PROTEIN.GRAPH.EDGE_CUTOFF = 10
_C.PROTEIN.GRAPH.NUM_KNN = 5
_C.PROTEIN.GRAPH.NUM_LAYER = 4
_C.PROTEIN.GRAPH.FC_BIAS = True

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 70
_C.SOLVER.BATCH_SIZE = 96
_C.SOLVER.NUM_WORKERS = 16 # number of workers for data loading
_C.SOLVER.LR = 5e-5
_C.SOLVER.SEED = 3076

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result_noname" # output directory
_C.RESULT.SAVE_MODEL = True

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 128
_C.DECODER.OUT_DIM = 64
_C.DECODER.BINARY = 1

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# data
_C.DATA = CN()
_C.DATA.TRAIN = "/share/home/grp-huangxd/chenyigang/DTI-Project/data/runs/20240921_114807/split/train_3d.parquet"
_C.DATA.VAL = "/share/home/grp-huangxd/chenyigang/DTI-Project/data/runs/20240921_114807/split/val_3d.parquet"
_C.DATA.TEST = "/share/home/grp-huangxd/chenyigang/DTI-Project/data/runs/20240921_114807/split/test_3d.parquet"

def get_cfg_defaults():
    return _C.clone()

