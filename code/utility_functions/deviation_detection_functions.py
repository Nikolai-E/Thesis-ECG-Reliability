import os
import pickle
import numpy as np
import itertools 
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from utility_functions.data_functions import *
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

t = 0.1 # Mahalanobis score 
ensemble_bins = np.linspace(0, 1, 11) # Bins for score histograms
outputfolder = '../output/'
experiment = 'exp1.1.1'
np.random.seed(42) # For reproducibility
method_names = ['maha', 'knn', 'iso', 'ocsvm', 'lof']




# Calculates distance of incomming feature using dist mu and inverse covariance matrix cov_inv
def compute_mahalanobis_distance(features, mu, cov_inv):
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    delta = features_clean - mu
    dist = np.sqrt(np.sum((delta @ cov_inv) * delta, axis=1))
    return np.nan_to_num(dist, nan=0.0, posinf=np.finfo(np.float32).max)

# coverts the maha distance into a score used as anomaly score from 0 to 1
# using refferance distrobution to scale it
def compute_mahalanobis_scores(distances, scaling):
    scores = 1.0 - np.exp(-t * distances / scaling)
    return np.nan_to_num(scores, nan=0.0)






# Fits the knn to a refference set
def fit_knn(features, n_neighbors=25):
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    n_samples = features_clean.shape[0]
    if n_samples == 0:
        raise ValueError("No input features knn")
    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(features_clean) 
    return knn 

# Uses the fitted knn to compute the distance score and anomoly score
def compute_knn_outlier_scores(knn, features):
    if features.size == 0: return np.array([])
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    distances, _ = knn.kneighbors(features_clean)
    scores = np.nanmean(distances, axis=1)
    return np.nan_to_num(scores, nan=0.0)






# Fits the IForest to a refference set.
def fit_isolation_forest(features, contamination='auto', random_state=42, n_estimators=200, max_samples=256):
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    if features_clean.shape[0] == 0:
        raise ValueError("No input features IForest")
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1, n_estimators=n_estimators, max_samples=max_samples)
    iso.fit(features_clean)
    return iso


# Uses the fitted IForest to compute the anomoly score 
def compute_isolation_forest_scores(iso, features):
    if features.size == 0: return np.array([])
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    # takes the negative such that anomalies get a higher score        
    scores = -iso.decision_function(features_clean)
    return np.nan_to_num(scores, nan=0.0)






# Fits the OCSVM to a refference set.
def fit_one_class_svm(features, nu=0.17):
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    if features_clean.shape[0] == 0:
        raise ValueError("No input features OCSVM")
    ocsvm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    ocsvm.fit(features_clean) 
    return ocsvm


# Uses the fitted OCSVM to compute the anomoly score 
def compute_ocsvm_scores(ocsvm, features):
    if features.size == 0: return np.array([])
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    # takes the negative such that anomalies get a higher score
    scores = -ocsvm.decision_function(features_clean)
    return np.nan_to_num(scores, nan=0.0)







# Fits the LOF to a refference set.
def fit_lof(features, n_neighbors=80, contamination='auto'):
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    n_samples = features_clean.shape[0]
    if n_samples == 0:
        raise ValueError("No input features LOF")
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination, n_jobs=-1)
    lof.fit(features_clean) 
    return lof 

# Uses the fitted LOF to compute the anomoly score 
def compute_lof_scores(lof, features):
    if features.size == 0: return np.array([])
    features_clean = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    scores = -lof.decision_function(features_clean)
    return np.nan_to_num(scores, nan=0.0)






# Scales the values, anomoly scores, between 0 and 1, based on refferance set
def min_max_scale(values, ref_values):
    if values.size == 0: return np.array([])
    ref_finite = ref_values[np.isfinite(ref_values)]
    if ref_finite.size == 0:
        raise ValueError("No input features")

    mi, ma = np.min(ref_finite), np.max(ref_finite)
    denominator = ma - mi

    values_clean = np.nan_to_num(values, nan=mi, posinf=ma, neginf=mi)
    scaled_values = (values_clean - mi) / denominator
    return np.clip(scaled_values, 0, 1)


# Calculates metrics, AUC
def calculate_metrics(y_true, y_scores):
    y_scores_clean = np.nan_to_num(y_scores, nan=0.0)
    roc_auc_val = roc_auc_score(y_true, y_scores_clean)
    return {'roc_auc': roc_auc_val}



# cross validates the weights
def cross_validate_weights(normal_scores_dict, anomaly_scores_dict, weights, n_folds=5):
    np.random.seed(42)

    n_normal = len(next(iter(val for val in normal_scores_dict.values() if len(val) > 0), []))
    n_anomaly = len(next(iter(val for val in anomaly_scores_dict.values() if len(val) > 0), []))
    normal_indices = np.random.permutation(n_normal)
    anomaly_indices = np.random.permutation(n_anomaly)

    fold_aucs = []
    for fold in range(n_folds):
        start_norm = fold * (n_normal // n_folds)
        end_norm = (fold + 1) * (n_normal // n_folds) if fold < n_folds - 1 else n_normal
        start_anom = fold * (n_anomaly // n_folds)
        end_anom = (fold + 1) * (n_anomaly // n_folds) if fold < n_folds - 1 else n_anomaly

        if start_norm >= end_norm and start_anom >= end_anom: continue

        val_normal_scores = {m: normal_scores_dict.get(m, np.array([]))[normal_indices[start_norm:end_norm]] for m in method_names}
        val_anomaly_scores = {m: anomaly_scores_dict.get(m, np.array([]))[anomaly_indices[start_anom:end_anom]] for m in method_names}

        norm_ens = compute_weighted_ensemble(val_normal_scores, weights)
        anom_ens = compute_weighted_ensemble(val_anomaly_scores, weights)

        if len(norm_ens) == 0 and len(anom_ens) == 0: continue

        y_true = np.concatenate([np.zeros_like(norm_ens), np.ones_like(anom_ens)])
        y_scores = np.concatenate([norm_ens, anom_ens])

        metrics = calculate_metrics(y_true, y_scores)
        fold_aucs.append(metrics['roc_auc']) 

    avg_auc = np.mean(fold_aucs) if fold_aucs else np.nan
    return {'roc_auc': avg_auc}


# Random search to find optimal weights
def random_search_cv(normal_scores_dict, anomaly_scores_dict, n_methods, cv=5, n_combinations=100000):
    print(f"Testing {n_combinations} random weights with cross validation")
    best_score = -1
    best_weights = np.ones(n_methods) / n_methods
    
    # make specific weights where 1 methode get whole weight
    specific_weights_to_test = []
    specific_weights_to_test.extend(list(np.eye(n_methods)))
    specific_weights_array = np.array(specific_weights_to_test)

    # random weights
    random_weight_combinations = np.random.dirichlet(np.ones(n_methods), size=n_combinations)

    # combine the weights
    weight_combinations = np.vstack((specific_weights_array, random_weight_combinations))
    
    # loop through and test weight combinations
    for weights in weight_combinations:
        cv_metrics = cross_validate_weights(normal_scores_dict, anomaly_scores_dict, weights, n_folds=cv)
        current_score = cv_metrics.get('roc_auc', np.nan)

        if np.isfinite(current_score) and current_score > best_score:
            best_score = current_score
            best_weights = weights

    best_metrics = {'roc_auc': best_score}
    return {'best_weights': best_weights, 'best_metrics': best_metrics}


# Print the weight optimization results, showing the best weights and the auc score
def print_random_search_results(results, feature_type):
    best_weights_final = results['best_weights']
    auc_score = results['best_metrics'].get('roc_auc', np.nan)

    print(f"\nBest{feature_type} weights:")
    print(f"Best CV auc: {auc_score:.4f}")
    print("Final methode weights")
    for i, method in enumerate(method_names):
        print(f"{method}: {best_weights_final[i]:.3f}")
    return best_weights_final

# Gives the optimal threshold using one of two approaches, gmean or set FPR
def find_optimal_threshold(normal_scores, anomaly_scores, method='gmean', target_fpr=0.05):
    y_true = np.concatenate([np.zeros_like(normal_scores), np.ones_like(anomaly_scores)])
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Maximise geometric mean
    if method == 'gmean':
        gmeans = np.sqrt(tpr * (1 - fpr + 1e-9))
        optimal_idx = np.argmax(gmeans)
        optimal_threshold = thresholds[optimal_idx]

    # Find threshold around set FPR
    elif method == 'fpr':
        optimal_idx = np.argmin(np.abs(fpr - target_fpr))
        optimal_threshold = thresholds[optimal_idx]   
        
    return optimal_threshold



# compute the weighted ensamble score using the score and weights
def compute_weighted_ensemble(scores_dict, weights):
    weights_sum = np.sum(weights)
    norm_weights = np.array(weights) / weights_sum if weights_sum > 0 else np.zeros_like(weights)
    num_samples = len(next(iter(val for val in scores_dict.values() if len(val) > 0), []))

    ensemble_scores = np.zeros(num_samples)
    for i, method in enumerate(method_names):
        method_scores = np.nan_to_num(scores_dict.get(method, np.array([])), nan=0.0)
        if len(method_scores) == num_samples:
             ensemble_scores += norm_weights[i] * method_scores
        elif len(method_scores) != 0: 
             raise ValueError("Error ensamble score")

    return ensemble_scores

