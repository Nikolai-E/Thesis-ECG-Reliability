import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from utils import utils
from models.fastai_model import fastai_model
import pandas as pd
import glob
import wfdb
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import itertools 

# Variables for importing model and PTB-XL data
sampling_frequency = 100
datafolder = '../data/ptbxl/'
outputfolder = '../output/'
task = 'superdiagnosticNORMandMI'
anomaly_task = 'superdiagnosticNotNORMandMI'
modelname = 'fastai_resnet1d_wang' ####### change model here: fastai_xresnet1d101, fastai_inception1d
num_classes = 2
input_shape = [1000, 12]
experiment = 'exp1.1.1'
mpath = f'../output/{experiment}/models/{modelname}/'
pretrainedfolder = mpath
new_wfdb_folder = os.path.join("..", "data", "wfdb15")
new_csv_file = os.path.join("..", "data", "exams.csv")
target_length = 1000
max_records = 10000
# Used for consistent output
np.random.seed(42)
torch.manual_seed(42)
os.makedirs(outputfolder, exist_ok=True)

t = 0.1 # Mahalanobis score 
ensemble_bins = np.linspace(0, 1, 11) # Bins for score histograms
outputfolder = '../output/'
experiment = 'exp1.1.1'
np.random.seed(42) # For reproducibility
method_names = ['maha', 'knn', 'iso', 'ocsvm', 'lof']


# used to process data to a specific frequency and length, here 100hz 10 second length
def preprocess_wfdb_signal(ecg_signal, orig_freq=400, target_freq=100, target_length=1000):
    # frequency
    if orig_freq != target_freq:
        num_samples = int(len(ecg_signal) * (target_freq / orig_freq))
        resampled = np.zeros((num_samples, ecg_signal.shape[1]))
        for lead in range(ecg_signal.shape[1]):
            resampled[:, lead] = signal.resample(ecg_signal[:, lead], num_samples)
        ecg_signal = resampled
    # length
    curr_length = ecg_signal.shape[0]
    if curr_length < target_length:
        padding = np.zeros((target_length - curr_length, ecg_signal.shape[1]))
        ecg_signal = np.vstack([ecg_signal, padding])
    elif curr_length > target_length:
        ecg_signal = ecg_signal[:target_length, :]
    return ecg_signal.astype(np.float32)


# Used to load a aditional dataset, code15, checks if folder and file exist

def load_custom_data(wfdb_folder, csv_file, max_records=10000):
    if not os.path.exists(wfdb_folder):
        raise FileNotFoundError("Custom data folder not found")
    if not os.path.exists(csv_file):
        raise FileNotFoundError("CSV file for custom data not found")

    df_csv = pd.read_csv(csv_file, dtype={'exam_id': str})
    if "exam_id" not in df_csv.columns:
         raise ValueError("Exam_id column not found")

    hea_files = glob.glob(os.path.join(wfdb_folder, "*.hea")) 
    if not hea_files:
        raise FileNotFoundError("No hea files found")
    record_ids_all = [os.path.splitext(os.path.basename(f))[0] for f in hea_files]

    signals = []
    valid_record_ids = []
    valid_csv_rows = []
    processed_rec_ids = set() 
    limit = min(max_records, len(record_ids_all))
    
    id_to_index_map = pd.Series(df_csv.index, index=df_csv['exam_id'].astype(str)).to_dict()

    for rec_id in tqdm(record_ids_all, desc="Loading custom dataset", disable=False):
        if len(valid_record_ids) >= limit:
            break
        if rec_id in processed_rec_ids:
            continue

            
            
        match_index = id_to_index_map.get(rec_id)
        if match_index is None:
            print(f"error record not found") 
            continue 

        rec_path = os.path.join(wfdb_folder, rec_id)
        record = wfdb.rdrecord(rec_path)
        signal_data = record.p_signal.astype(np.float32)
        processed_signal = preprocess_wfdb_signal(
            signal_data, record.fs, sampling_frequency, target_length
        )

        signals.append(processed_signal)
        valid_record_ids.append(rec_id)
        valid_csv_rows.append(df_csv.iloc[match_index])
        processed_rec_ids.add(rec_id)


    matched_df = pd.DataFrame(valid_csv_rows).reset_index(drop=True)
    print(f"Loaded {len(signals)} records")
    return signals, valid_record_ids, matched_df


# Creates syntethic anomolies for testing, uses a input ECG as base
def create_synthetic_anomalies(original_ecg):
    anom_ecg_1 = original_ecg.copy()
    anom_ecg_1[:200, :] = 1
    anom_ecg_1[-200:, :] = 1
    
    anom_ecg_2 = original_ecg.copy()
    anom_ecg_2[:400, :] = 0.0
    
    anom_ecg_3 = original_ecg.copy()
    anom_ecg_3 += 1
    
    anom_ecg_4 = original_ecg.copy()
    num_samples = anom_ecg_4.shape[0]
    drift = np.linspace(0, 1, num_samples)[:,None]
    anom_ecg_4 += drift
    
    anom_ecg_5 = original_ecg.copy()
    num_samples = anom_ecg_5.shape[0]
    rng = np.random.default_rng(42)
    for _ in range(3): 
        start = rng.integers(0, num_samples - 60)
        anom_ecg_5[start:start+60] += rng.normal(0, 0.4, size=(60, anom_ecg_5.shape[1]))
    
    anom_ecg_6 = original_ecg.copy()
    idx = 0
    while idx < anom_ecg_6.shape[0]: 
        anom_ecg_6[idx:min(idx+2, anom_ecg_6.shape[0]), :] += 2.0
        idx += 50
        
    return np.stack([anom_ecg_1, anom_ecg_2, anom_ecg_3, anom_ecg_4, anom_ecg_5, anom_ecg_6], axis=0) 






# Extracts features from ECGs, both deep learning and time-domain

def extract_features(ecg_array, pytorch_model, standard_scaler, device, layer_features):
    out_features = []
    time_domain_features = []

    for ecg in tqdm(ecg_array, desc="Extracting features", disable=False):
        layer_features.clear() 

        # deep learning features
        ecg_std = utils.apply_standardizer(np.expand_dims(ecg, axis=0), standard_scaler)
        tensor = torch.tensor(ecg_std, dtype=torch.float32).to(device)
        tensor = tensor.permute(0, 2, 1)

        with torch.no_grad():
            _ = pytorch_model(tensor)
        
        # save features, squeeze only removes some uneeded data
        dl_features = layer_features["Batch_norm"].squeeze()

        # time-domain features
        td_features = []
        for lead_idx in range(ecg.shape[1]):
            lead_signal = ecg[:, lead_idx].astype(np.float64) 
            mean = np.mean(lead_signal)
            std_dev = np.std(lead_signal)
            skewness = stats.skew(lead_signal)
            kurtosis = stats.kurtosis(lead_signal)
            signal_range = np.ptp(lead_signal) 
            energy = np.sum(lead_signal**2)
            lead_features = np.array([mean, std_dev, skewness, kurtosis, signal_range, energy])
            td_features.extend(lead_features)
            
            
        time_domain_features.append(np.nan_to_num(np.array(td_features), nan=0.0, posinf=0.0, neginf=0.0))
        out_features.append(dl_features)
    return np.array(out_features), np.array(time_domain_features)


# Used to either load or extract features, avoiding repeated extraction when rerunning code
def load_or_extract(filename, data_array, td_filename=None, pytorch_model=None,
                    standard_scaler=None, device=None, layer_features=None):
    # load features
    if os.path.exists(filename) and (td_filename is None or os.path.exists(td_filename)):
        with open(filename, 'rb') as f: feats = pickle.load(f)
        print("Loaded DL features")
        td_feats = None
        if td_filename is not None:
            with open(td_filename, 'rb') as f: td_feats = pickle.load(f)
            print("Loaded TD features")
        return np.array(feats), np.array(td_feats) if td_feats is not None else None
    else:
        # extract features
        print("Extracting features") 
        feats, td_feats = extract_features(data_array, pytorch_model, standard_scaler, device, layer_features)
        # save features
        with open(filename, 'wb') as f: pickle.dump(feats, f)
        print("Saved fetures")
        if td_filename is not None:
            with open(td_filename, 'wb') as f: pickle.dump(td_feats, f)
        return feats, td_feats
    
    
    
    
    
    
    


# Loads features from cell 1, all extracted and preprocessed.
# makes a dictionary used to loop through all data
def load_features():
    print("Loading features")
    features = {}
    dataset_keys_map = {
        'dl_normal_ref': f'{experiment}_feats_normal_ref_norm.pkl',
        'td_normal_ref': f'{experiment}_td_feats_normal_ref_z.pkl',
        'dl_norm_val': f'{experiment}_feats_norm_val_norm.pkl',
        'td_norm_val': f'{experiment}_td_feats_norm_val_z.pkl',
        'dl_notnorm_val': f'{experiment}_feats_notnorm_val_norm.pkl',
        'td_notnorm_val': f'{experiment}_td_feats_notnorm_val_z.pkl',
        'dl_norm_test': f'{experiment}_feats_norm_test_norm.pkl',
        'td_norm_test': f'{experiment}_td_feats_norm_test_z.pkl',
        'dl_notnorm_test': f'{experiment}_feats_notnorm_test_norm.pkl',
        'td_notnorm_test': f'{experiment}_td_feats_notnorm_test_z.pkl',
        'dl_anomalies_synth': f'{experiment}_feats_anomalies_synth_norm.pkl', 
        'td_anomalies_synth': f'{experiment}_td_feats_anomalies_synth_z.pkl',
        'dl_new': f'{experiment}_feats_new_norm.pkl',
        'td_new': f'{experiment}_td_feats_new_z.pkl',
    }

    for key, filename in dataset_keys_map.items():
        filepath = os.path.join(outputfolder, filename)
        optional = 'synth' in key or 'new' in key

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                features[key] = pickle.load(f)
        elif not optional:
             raise FileNotFoundError(f"File not found = {filepath}")
        else:
             features[key] = np.array([])

    # Load New Dataset extra files
    new_record_ids_path = os.path.join(outputfolder, f'{experiment}_new_record_ids.pkl')
    new_diag_path = os.path.join(outputfolder, f'{experiment}_new_diagnostic_labels.pkl')
    features['new_record_ids'] = None
    features['diagnostic_data'] = None
    if os.path.exists(new_record_ids_path):
        with open(new_record_ids_path, 'rb') as f: features['new_record_ids'] = pickle.load(f)
    if os.path.exists(new_diag_path):
        with open(new_diag_path, 'rb') as f: features['diagnostic_data'] = pickle.load(f)

    
    print(f"Training data / refference set: {len(features.get('dl_normal_ref', []))}")
    print(f"Validation set: {len(features.get('dl_norm_val', []))} Norm/MI, {len(features.get('dl_notnorm_val', []))} Not Norm/MI")
    print(f"Test Set {len(features.get('dl_norm_test', []))} Norm/MI, {len(features.get('dl_notnorm_test', []))} Not Norm/MI")
    print(f"Syntethic anomalies: {len(features.get('dl_anomalies_synth', []))}") 
    print(f"New/additional data set: {len(features.get('dl_new', []))}")
    return features
    
    

