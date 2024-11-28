import rdkit.Chem
from tqdm import tqdm
import numpy as np
import pickle as pkl
import torch
import torch_geometric
import torch_cluster
import torch.utils.data as data
from dgllife.utils import CanonicalAtomFeaturizer
from utils import _rbf, _normalize, match_feature_with_key, print_with_time
from graph_utils import build_heterograph
import os 
import asyncio
from tqdm import tqdm

def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v

class ConditionalInitiableDataset(data.Dataset):
    """
    The father object of all datasets. Used to store configurations and preprocessed data.
    """
    def __init__(self, **config):
        self.config = config
        self.k_p = config["PROTEIN"]["GRAPH"]["NUM_KNN"]
        self.r_p = config["PROTEIN"]["GRAPH"]["EDGE_CUTOFF"]
        self.l_p = config["PROTEIN"]["MAX_LENGTH"]
        self.ec_d = config["DRUG"]["EDGE_CUTOFF"]
        self.nr_d = config["DRUG"]["NUM_RDF"]
        self.protein_graph_path = config["PROTEIN"]["GRAPH"]["PATH"] # example: "PATH/TO/DATA/protein_graph.pkl"
        self.protein_graph_path = os.path.abspath(self.protein_graph_path)[:-4] + \
            f"_k{self.k_p}_r{self.r_p}_l{self.l_p}.pkl" # auto generated path
        if os.path.exists(self.protein_graph_path):
            with open(self.protein_graph_path, "rb") as f:
                self.protein_data_dict = pkl.load(f)
        else:
            self.protein_data_dict = {}
            if not os.path.exists(config["PROTEIN"]["GRAPH"]["PATH"]):
                print_with_time(f"Protein graph file not found. Please check the path at {config['PROTEIN']['GRAPH']['PATH']}")
                print_with_time("Preprocessing is recommended. Run this script with the same config file to preprocess protein data.")
                print_with_time("Example: \npython dataloader.py config_yaml/default.yaml")
                print_with_time("The training may be slower than expected...")
        self.protein_crood_path = config["PROTEIN"]["GRAPH"]["COORD_PATH"]
        if not os.path.exists(self.protein_crood_path):
            raise FileNotFoundError(f"Protein coordinate file not found at {self.protein_crood_path}")
        else:
            with open(self.protein_crood_path, "rb") as f:
                self.coords_df = pkl.load(f)
        self.drug_data_dict = {}

    def featurize_protein(self, uniprot_id, k, r, l, name=None):
        """
        Parameters
        ----------
        uniprot_id: str
            Uniprot ID of protein
        k: int
            Number of nearest neighbors for KNN.
        r: float
            Radius for R_SPHERE edges.
        l: int
            Max length of protein sequence.

        Returns
        -------
        graph: dgl graph
            A heterogeneous graph with KNN and R_SPHERE edges.
        """ 
        if uniprot_id in self.protein_data_dict.keys():
            return self.protein_data_dict[uniprot_id]
        seq = self.coords_df.loc[uniprot_id, "sequence"]
        seq = seq.upper()
        seq = match_feature_with_key(seq)
        coord = self.coords_df.loc[uniprot_id, "crod"] # np.array of shape (L, 3)

        g = build_heterograph(
            coord, seq, 
            k=k, r_sphere=r, max_len=l
        )
        self.protein_data_dict[uniprot_id] = g
        return g
    
    def featurize_drug(self, sdf, id=None, name=None, edge_cutoff=4.5, num_rbf=16) -> torch_geometric.data.Data:
        """
        Parameters
        ----------
        sdf_path: str
            Path to sdf file
        name: str
            Name of drug
        Returns
        -------
        graph: torch_geometric.data.Data
            A torch_geometric graph
        """
        if id is not None:
            if id in self.drug_data_dict.keys():
                return self.drug_data_dict[id]

        mol = rdkit.Chem.MolFromMolBlock(sdf)
        conf = mol.GetConformer()
        with torch.no_grad():
            coords = conf.GetPositions()
            coords = torch.as_tensor(coords, dtype=torch.float32)

            atom_featurizer = CanonicalAtomFeaturizer()
            
            atom_feature = atom_featurizer(mol)["h"]  
            edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

        node_s = atom_feature
        node_v = coords.unsqueeze(1)
        edge_s, edge_v = _build_edge_feature(
            coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

        data = torch_geometric.data.Data(
            x=coords, edge_index=edge_index, name=name,
            node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)

        if id is not None:
            self.drug_data_dict[id] = data

        return data

    def _preprocessing(self):
        """
        Used to genrerate protein graph data and save it to disk
        """
        print_with_time("Preprocessing protein data")
        if os.path.exists(self.protein_graph_path):
            print_with_time("Protein graph data already exists")
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=40) as executor:
                tasks = []
                results = []
                for uniprot_id in tqdm(self.coords_df.index):
                    tasks.append(executor.submit(\
                        self.featurize_protein, uniprot_id, self.k_p, self.r_p, self.l_p))
                print_with_time("Waiting for tasks to complete")
                for task in tqdm(tasks):
                    result = task.result()
                    results.append(result)
            
            self.protein_data_dict = dict(zip(self.coords_df.index, results))

            with open(self.protein_graph_path, "wb") as f:
                pkl.dump(self.protein_data_dict, f)
            print_with_time(f"Protein graph data saved to {self.protein_graph_path}")

class DTIDataset(ConditionalInitiableDataset):
    def __init__(self, list_IDs, df, father):
        print_with_time("Recieved data, length:", len(df))
        self.df = df[df['target_uniprot_id']\
                .isin(father.coords_df.index)]\
            .reset_index(drop=True)
        list_IDs = [i for i in list_IDs if i in df.index]
        self.list_IDs = list_IDs
        self.father = father
        print_with_time("Loaded data, length:", len(self.df))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        row = self.df.iloc[index]
        v_d = row['sdf']
        v_d = self.father.featurize_drug(v_d, id=row['drug_chembl_id'])

        v_p = row['target_uniprot_id']
        v_p = self.father.featurize_protein(v_p, self.father.k_p, self.father.r_p, self.father.l_p)

        y = row["label"]

        return v_d, v_p, y

if __name__ == "__main__":
    """
    Run this script to generate protein graph (preprocessing).
    example: python dataloader.py config_yaml/default.yaml
    """
    from configs import get_cfg_defaults
    import sys
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
    print_with_time(dict(cfg))

    dataset = ConditionalInitiableDataset(**cfg)

    dataset._preprocessing()
