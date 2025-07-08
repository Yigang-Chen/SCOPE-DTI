import os
import random
import numpy as np
import torch
from torch_geometric.data import Batch
import torch_geometric
import logging
import dgl

all_for_assign = np.array([
[3.854415274463007246e-01,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,3.701467387488588390e-01,1.843457636052180382e-01],
[2.684964200477326091e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,3.701467387488588390e-01,1.843457636052180382e-01],
[0.000000000000000000e+00,6.666666666666666297e-01,5.000000000000000000e-01,1.000000000000000000e+00,3.333333333333333703e-01,2.103840483044302490e-01,6.758466201080510771e-01],
[5.369928400954654402e-02,6.666666666666666297e-01,5.000000000000000000e-01,1.000000000000000000e+00,3.333333333333333703e-01,3.473285122516323042e-01,6.758466201080510771e-01],
[3.233890214797136564e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,8.310398090289965456e-01,1.843457636052180382e-01],
[3.818615751789976032e-01,3.333333333333333148e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,2.653935266446673658e-01,1.843457636052180382e-01],
[5.608591885441527314e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,3.827845257319384409e-01,5.622611674792461489e-01],
[3.782816229116945927e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,7.620234501158462681e-01,1.843457636052181492e-01],
[8.138424821002385734e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,4.400758267218984887e-01,5.272104361575966625e-01],
[3.830548926014320510e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,7.620234501158462681e-01,1.843457636052181492e-01],
[3.544152744630071905e-01,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,6.591659060591167352e-01,1.843457636052180382e-01],
[3.150357995226730767e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,0.000000000000000000e+00,7.521412570826193633e-01],
[4.212410501193317169e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,5.438812048023590195e-01,0.000000000000000000e+00],
[3.436754176610978262e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,1.369444639472022773e-01,7.521412570826193633e-01],
[1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,9.999999999999998890e-01,1.361616232535282078e-01,1.000000000000000000e+00],
[3.472553699284009476e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,4.103770273116591483e-02,4.509157991830281542e-01],
[3.424821002386633784e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,1.774204872568978519e-01,4.509157991830281542e-01],
[3.806682577565632664e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,6.250789861686442128e-01,1.843457636052181492e-01],
[3.723150357995225757e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,3.333333333333333703e-01,9.999999999999998890e-01,3.924100672025299108e-01],
[3.448687350835322740e-01,3.333333333333333148e-01,0.000000000000000000e+00,1.000000000000000000e+00,3.333333333333333703e-01,7.276907954784807009e-01,4.509157991830281542e-01],
]) # FORM MAPE-PPI

def match_feature_with_key(x, all_for_assign=all_for_assign):
    key_dict = {
        'A': all_for_assign[0, :],  # ALA
        'C': all_for_assign[1, :],  # CYS
        'D': all_for_assign[2, :],  # ASP
        'E': all_for_assign[3, :],  # GLU
        'F': all_for_assign[4, :],  # PHE
        'G': all_for_assign[5, :],  # GLY
        'H': all_for_assign[6, :],  # HIS
        'I': all_for_assign[7, :],  # ILE
        'K': all_for_assign[8, :],  # LYS
        'L': all_for_assign[9, :],  # LEU
        'M': all_for_assign[10, :], # MET
        'N': all_for_assign[11, :], # ASN
        'P': all_for_assign[12, :], # PRO
        'Q': all_for_assign[13, :], # GLN
        'R': all_for_assign[14, :], # ARG
        'S': all_for_assign[15, :], # SER
        'T': all_for_assign[16, :], # THR
        'V': all_for_assign[17, :], # VAL
        'W': all_for_assign[18, :], # TRP
        'Y': all_for_assign[19, :], # TYR
    }
    
    x_p = np.zeros((len(x), 7))
    
    for j in range(len(x)):
        x_p[j] = key_dict[x[j]]
        
    return x_p



def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(batch):
    d, p, y = zip(*batch)
    d = Batch.from_data_list(d)
    p = dgl.batch(p)
    y = torch.tensor(y, dtype=torch.float32)
    return d, p, y


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

import datetime 
def print_with_time(*arg,**args):
    print(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), *arg,**args)

# ============= Protein Sequence Encoding =============

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

def integer_label_protein(sequence, max_length=2000):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as padding."
            )
    return encoding

def sequence_collate_func(batch):
    """
    Collate function for sequence mode
    """
    d, p, y = zip(*batch)
    d = Batch.from_data_list(d)
    p = torch.stack(p)
    y = torch.tensor(y, dtype=torch.float32)
    return d, p, y

def get_collate_func(protein_mode):
    """
    Get appropriate collate function based on protein mode
    """
    if protein_mode == "graph":
        return graph_collate_func
    elif protein_mode == "sequence":
        return sequence_collate_func
    else:
        raise ValueError(f"Unsupported protein mode: {protein_mode}")

def validate_protein_mode_config(config):
    """
    Validate configuration for protein mode
    """
    mode = config.get("PROTEIN", {}).get("MODE", "graph")
    if mode not in ["graph", "sequence"]:
        raise ValueError(f"Invalid protein mode: {mode}. Must be 'graph' or 'sequence'")
    
    if mode == "graph":
        coord_path = config.get("PROTEIN", {}).get("GRAPH", {}).get("COORD_PATH", "")
        if not coord_path or not os.path.exists(coord_path):
            raise FileNotFoundError(f"Protein coordinate file not found: {coord_path}")
    
    return mode

# ============= Data Processing =============
# reference version
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def calculate_stats(df, name):
    total_count = len(df)
    label_0_count = len(df[df['label'] == 0])
    label_1_count = len(df[df['label'] == 1])
    unique_smiles_count = df['smiles'].nunique()
    unique_sequence_count = df['sequence'].nunique()
    
    return {
        "data": name,
        "total_dti": total_count,
        "label 0": label_0_count,
        "label 0 ratio": f"{label_0_count / total_count:.2%}",
        "label 1": label_1_count,
        "label 1 ratio": f"{label_1_count / total_count:.2%}",
        "unique_smiles": unique_smiles_count,
        "unique_sequence": unique_sequence_count
    }
def split_dataset_by_smiles(data_df, save_path, test_size=0.2, val_size=0.1):
    # Step 1: Split SMILES into training, validation, and test sets
    all_smiles = data_df['smiles'].unique()
    train_smiles, temp_smiles = train_test_split(all_smiles, test_size=(test_size + val_size))
    val_smiles, test_smiles = train_test_split(temp_smiles, test_size=test_size / (test_size + val_size))

    # Step 2: Create dataframes for training, validation, and test sets based on SMILES
    train_df = data_df[data_df['smiles'].isin(train_smiles)]
    val_df = data_df[data_df['smiles'].isin(val_smiles)]
    test_df = data_df[data_df['smiles'].isin(test_smiles)]

    # Step 3: Ensure all sequences in val and test are also present in train
    val_sequences = set(val_df['sequence'])
    test_sequences = set(test_df['sequence'])

    # Find sequences missing in train
    missing_sequences = (val_sequences | test_sequences) - set(train_df['sequence'])
    
    if missing_sequences:
        # Add rows with missing sequences to train_df
        additional_rows = data_df[data_df['sequence'].isin(missing_sequences)]
        train_df = pd.concat([train_df, additional_rows])

    # Ensure no SMILES in val and test appear in train
    train_smiles_set = set(train_df['smiles'])
    val_df = val_df[~val_df['smiles'].isin(train_smiles_set)]
    test_df = test_df[~test_df['smiles'].isin(train_smiles_set)]
    
    # Validation step to ensure the conditions are met
    train_sequences_set = set(train_df['sequence'])
    
    if not val_sequences.issubset(train_sequences_set):
        raise ValueError("Validation sequences are not a subset of training sequences.")
    
    if not test_sequences.issubset(train_sequences_set):
        raise ValueError("Test sequences are not a subset of training sequences.")
    
    if len(train_smiles_set.intersection(val_df['smiles'])) > 0:
        raise ValueError("Some SMILES in the validation set are also in the training set.")
    
    if len(train_smiles_set.intersection(test_df['smiles'])) > 0:
        raise ValueError("Some SMILES in the test set are also in the training set.")
    
    # 计算每个数据集的统计信息
    origin_stats = calculate_stats(data_df, "Origin")
    train_stats = calculate_stats(train_df, "Train")
    val_stats = calculate_stats(val_df, "Validation")
    test_stats = calculate_stats(test_df, "Test")

    # 创建 DataFrame 并显示
    stats_df = pd.DataFrame([origin_stats, train_stats, val_stats, test_stats])
    print(stats_df.to_string(index=False))

    train_df.to_parquet(os.path.join(save_path, "train.parquet"))
    val_df.to_parquet(os.path.join(save_path, "val.parquet"))
    test_df.to_parquet(os.path.join(save_path, "test.parquet"))
    stats_df.to_csv(os.path.join(save_path, "stats.csv"))

def data_filter(data_df, n, label_ratio=0.7):
    # Group the DataFrame by 'sequence' and 'label' and count unique 'smiles' for each
    sequence_counts = data_df.groupby(['sequence', 'label'])['smiles'].nunique().reset_index()
    sequence_counts.columns = ['sequence', 'label', 'smiles_count']
    
    # Pivot the table to get separate columns for each label's smiles count
    sequence_pivot = sequence_counts.pivot(index='sequence', columns='label', values='smiles_count').fillna(0)
    sequence_pivot.columns = ['smiles_count_label_0', 'smiles_count_label_1']
    
    # Calculate the total smiles count for each sequence
    sequence_pivot['total_smiles_count'] = sequence_pivot['smiles_count_label_0'] + sequence_pivot['smiles_count_label_1']
    
    # Calculate the proportion of smiles for each label
    sequence_pivot['ratio_label_0'] = sequence_pivot['smiles_count_label_0'] / sequence_pivot['total_smiles_count']
    sequence_pivot['ratio_label_1'] = sequence_pivot['smiles_count_label_1'] / sequence_pivot['total_smiles_count']
    
    # Classify sequences based on the ratio
    sequence_pivot['category'] = 'Not Predominantly One Label'
    sequence_pivot.loc[(sequence_pivot['ratio_label_0'] > label_ratio) | 
                       (sequence_pivot['ratio_label_1'] > label_ratio), 'category'] = 'Predominantly One Label'
    
    # Attempt to rescue 'Predominantly One Label' sequences by lowering their ratio
    rescue_sequences = sequence_pivot[sequence_pivot['category'] == 'Predominantly One Label'].index
    rescued_data = []
    
    for sequence in tqdm(rescue_sequences, desc="Rescuing sequences"):
        sequence_data = data_df[data_df['sequence'] == sequence]
        
        # Calculate current ratios
        label_0_count = sequence_data[sequence_data['label'] == 0].shape[0]
        label_1_count = sequence_data[sequence_data['label'] == 1].shape[0]
        total_count = label_0_count + label_1_count
        
        # Determine which label is dominant
        if label_0_count / total_count > label_ratio:
            max_label = 0
            max_count = label_0_count
            min_count = label_1_count
        else:
            max_label = 1
            max_count = label_1_count
            min_count = label_0_count
        
        # Calculate the number to keep to meet the ratio requirement
        target_count = min_count * 2
        number_to_remove = max_count - target_count
        
        # Randomly sample data to remove to meet the label ratio
        if number_to_remove > 0:
            data_to_keep = pd.concat([
                sequence_data[sequence_data['label'] != max_label],
                sequence_data[sequence_data['label'] == max_label].sample(n=max_count - number_to_remove, random_state=42)
            ])
        else:
            data_to_keep = sequence_data
        
        # Check if adjustment was successful
        adjusted_ratio = data_to_keep['label'].value_counts(normalize=True).max()
        if adjusted_ratio <= label_ratio:
            rescued_data.append(data_to_keep)
    
    # Combine rescued data back into original dataframe
    if rescued_data:
        rescued_data_df = pd.concat(rescued_data)
    else:
        rescued_data_df = pd.DataFrame(columns=data_df.columns)

    # Filter 'Not Predominantly One Label' sequences
    not_predominantly_one_label_sequences = sequence_pivot[sequence_pivot['category'] == 'Not Predominantly One Label'].index
    filtered_data_df = data_df[data_df['sequence'].isin(not_predominantly_one_label_sequences)]
    
    # Combine with rescued sequences
    filtered_data_df = pd.concat([filtered_data_df, rescued_data_df])
    
    # Further filter to remove sequences with less than n unique smiles
    sequence_smiles_counts = filtered_data_df.groupby('sequence')['smiles'].nunique().reset_index()
    sequence_smiles_counts = sequence_smiles_counts[sequence_smiles_counts['smiles'] >= n]
    
    # Keep only the sequences that have at least n unique smiles
    valid_sequences = sequence_smiles_counts['sequence']
    final_filtered_df = filtered_data_df[filtered_data_df['sequence'].isin(valid_sequences)]

    return final_filtered_df