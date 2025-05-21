#!/usr/bin/env python
# coding: utf-8

# # Train and use model

# In[1]:


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[2]:


from experiments.scp_experiment import SCP_Experiment
from utils import utils
from configs.fastai_configs import *
from configs.wavelet_configs import *
import torch
import multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# config for data, output, model and experiment used during training
data = '../data/ptbxl/'
output = '../output/'
models = [conf_fastai_xresnet1d101] # conf_fastai_resnet1d_wang , conf_fastai_inception1d
experiments = [('exp1.1.1', 'superdiagnosticNORMandMI')]


# Perform preprocessing, training and evaluation to train the model
for name, task in experiments:
    e = SCP_Experiment(name, task, data, output, models)
    e.prepare()
    e.perform()
    e.evaluate()

# # Generate AUC table
utils.generate_ptbxl_summary_table()



# In[1]:


from utils import utils
from models.fastai_model import fastai_model
import pickle
import numpy as np
from scipy.special import softmax

# Model parameters
sampling_frequency = 100
datafolder = '../data/ptbxl/'
task = 'superdiagnosticNORMandMI'
outputfolder = '../output/'
modelname = 'fastai_xresnet1d101'
num_classes = 2  
input_shape = [1000, 12]
experiment = 'exp1.1.1'  
mpath = f'../output/{experiment}/models/{modelname}/'
pretrainedfolder = mpath

# Load the MI and NORM data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
data, labels, Y, _ = utils.select_data(
    data, labels, task, min_samples=0, outputfolder=outputfolder
)



i = 155  # ECG to use
ecg_sample = data[i]  

# Label info
label_info = labels.iloc[i]

# Print label info
print(f"ECG index: {i}")
print(f"Patient ID: {label_info['patient_id']}")
print(f"Filename: {label_info['filename_lr']}")
print(f"Recording Date: {label_info['recording_date']}")
print(f"scp codes: {label_info['scp_codes']}")

# Load standardizer for preprocessing and preprocess ECG
standard_scaler = pickle.load(
    open(f'../output/{experiment}/data/standard_scaler.pkl', "rb")
)
ecg_sample_batch = np.expand_dims(ecg_sample, axis=0) 
ecg_sample_standardized = utils.apply_standardizer(
    ecg_sample_batch, standard_scaler
)


# Load model
model = fastai_model(
    modelname,
    num_classes,
    sampling_frequency,
    mpath,
    input_shape=input_shape,
    pretrainedfolder=pretrainedfolder,
    n_classes_pretrained=num_classes,
    pretrained=True,
    epochs_finetuning=0,
)

# Make prediction
y_pred = model.predict(ecg_sample_standardized)

print(y_pred)

# Make probabilities
probabilities = softmax(y_pred, axis=1)
print(f"Prediction: {probabilities}")


predicted_class_index = np.argmax(probabilities, axis=1)[0]
class_names = ['MI', 'Normal']
predicted_class_name = class_names[predicted_class_index]
print(f"Predicted class name: {predicted_class_name}")


# # ECG visualization

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# example ECG
normal_index = 155 
ecg_sample_normal = data[normal_index]

# Time axis
fs = sampling_frequency 
t = np.arange(ecg_sample_normal.shape[0]) / fs 

# Lead names
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Create plot grid
fig, axs = plt.subplots(6, 2, figsize=(18, 20)) 
axs = axs.flatten()


# Plot each lead
for i in range(12):
    axs[i].plot(t, ecg_sample_normal[:, i], linewidth=1.5, label='Normal ECG Waveforms')
    axs[i].set_title(f'Lead {leads[i]}', fontsize=14, pad=10) 
    axs[i].grid(True, linestyle=':', linewidth=0.8)
    axs[i].tick_params(axis='both', which='major', labelsize=10)

    # Only label left side
    if i % 2 == 0:
        axs[i].set_ylabel('Amplitude (mV)', fontsize=12)
    else:
        axs[i].set_yticklabels([])  

    # Only label bottom 
    if i >= 10: 
        axs[i].set_xlabel('Time (s)', fontsize=12)
    else:
        axs[i].set_xticklabels([]) 

# Add a legend label
handles, labels = axs[0].get_legend_handles_labels() 
fig.legend(handles, labels, loc='upper center', fontsize=14, frameon=True, ncol=1, bbox_to_anchor=(0.5, 0.96))

# adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.subplots_adjust(wspace=0.05, hspace=0.15)  

# show plot
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals import ecg

# Lead index
lead_indices = {
    'I': 0,
    'II': 1,
    'III': 2,
    'aVR': 3,
    'aVL': 4,
    'aVF': 5,
    'V1': 6,
    'V2': 7,
    'V3': 8,
    'V4': 9,
    'V5': 10,
    'V6': 11
}

# Selected leads
all_leads_names = ['II', 'III', 'aVF','I', 'aVL']
all_leads = [lead_indices[lead] for lead in all_leads_names]

# Select MI
mi_index = 8929 

# Load ECG samples for MI only
ecg_sample_mi = data[mi_index]

# Extract beats to inspect
def extract_beats(signal, sampling_frequency):
    out = ecg.ecg(signal=signal, sampling_rate=sampling_frequency, show=False)
    r_peaks = out['rpeaks']
    beats = []
    beat_duration = 0.8  
    samples_per_beat = int(beat_duration * sampling_frequency / 2)  
    for r in r_peaks:
        start = max(r - samples_per_beat, 0)
        end = min(r + samples_per_beat, len(signal))
        beat = signal[start:end]
        if len(beat) == 2 * samples_per_beat:
            beats.append(beat)
    return np.array(beats)


# Set fonts
plt.rc('font', size=18)     
plt.rc('axes', titlesize=20)      
plt.rc('axes', labelsize=18)    
plt.rc('xtick', labelsize=16)  
plt.rc('ytick', labelsize=16)     
plt.rc('legend', fontsize=18)    

# Create plots for each beat
fig, axs = plt.subplots(len(all_leads), 1, figsize=(12, 20), sharex=True)


# Pick the first lead and plot it
for idx, lead in enumerate(all_leads):
    lead_name = all_leads_names[idx]
    signal_mi = ecg_sample_mi[:, lead]
    beats_mi = extract_beats(signal_mi, sampling_frequency)
    beat_mi = beats_mi[0]
    t_beat = np.linspace(-0.4, 0.4, len(beat_mi)) 
    ax = axs[idx]
    ax.plot(t_beat, beat_mi, linewidth=2.5, color='#4f86c6', label='MI Beat')
    # titles
    ax.set_title(f'Lead {lead_name}', pad=15)
    ax.set_ylabel('Amplitude (mV)', labelpad=10)
    ax.grid(True, linestyle=':', linewidth=1)

    # Highlight ST segment, yellow section
    st_start = 0.04  
    st_end = 0.2     
    ax.axvspan(st_start, st_end, color='yellow', alpha=0.3, label='ST Segment')

    # Show baseline, 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    ax.text(t_beat[0], -0.15, 'Baseline (0 mV)', fontsize=16, color='black')
    y_min = beat_mi.min() - 0.1
    y_max = beat_mi.max() + 0.1
    ax.set_ylim([y_min, y_max])

    
# Add legend text
handles, labels = axs[0].get_legend_handles_labels()  
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), fontsize=18, frameon=True, ncol=3)
axs[-1].set_xlabel('Time (s)', labelpad=10)
plt.tight_layout(rect=[0, 0, 1, 0.9]) 

# Show the beats
plt.show()


# # Model architecture and feature vector

# In[16]:


from utils import utils
from models.fastai_model import fastai_model
import pickle
import numpy as np
from scipy.special import softmax
import torch

# Model parameters
sampling_frequency = 100
datafolder = '../data/ptbxl/'
task = 'superdiagnosticNORMandMI'
outputfolder = '../output/'
modelname = 'fastai_xresnet1d101'
num_classes = 2  
input_shape = [1000, 12]
experiment = 'exp1.1.1'  
mpath = f'../output/{experiment}/models/{modelname}/'
pretrainedfolder = mpath

# Load the MI and NORM data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
data, labels, Y, _ = utils.select_data(
    data, labels, task, min_samples=0, outputfolder=outputfolder
)

i = 200  # ECG index
ecg_sample = data[i]
label_info = labels.iloc[i]




# Load standardizer for preprocessing and preprocess ECG
standard_scaler = pickle.load(
    open(f'../output/{experiment}/data/standard_scaler.pkl', "rb")
)
ecg_sample_batch = np.expand_dims(ecg_sample, axis=0) 
ecg_sample_standardized = utils.apply_standardizer(
    ecg_sample_batch, standard_scaler
)


# Load model
model = fastai_model(
    modelname,
    num_classes,
    sampling_frequency,
    mpath,
    input_shape=input_shape,
    pretrainedfolder=pretrainedfolder,
    n_classes_pretrained=num_classes,
    pretrained=True,
    epochs_finetuning=0,
)


# Load model using dummy data to get the underlying model to inspect arcitecture and exract feature
X = [ecg_sample_standardized[0]]
y_dummy = [0]  
learn = model._get_learner(X, y_dummy, X, y_dummy)
learn.load(model.name)
pytorch_model = learn.model
pytorch_model.eval()

# Attach hook
features = []
def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())

# Inspect model
print(pytorch_model)
print("\nModel Head structure")
print(pytorch_model[-1])

# Attach the hook to the head layer
hook = pytorch_model[-1][-3].register_forward_hook(hook_fn)


ecg_sample_tensor = torch.tensor(ecg_sample_standardized, dtype=torch.float32)
ecg_sample_tensor = ecg_sample_tensor.permute(0, 2, 1)
ecg_sample_tensor = ecg_sample_tensor.to(learn.data.device)

# Run the model on the ecg
with torch.no_grad():
    outputs = pytorch_model(ecg_sample_tensor)

hook.remove()


# In[17]:


import numpy as np

# Print label info
print(f"ECG index: {i}")
print(f"Patient ID: {label_info['patient_id']}")
print(f"Filename: {label_info['filename_lr']}")
print(f"Recording Date: {label_info['recording_date']}")
print(f"scp codes: {label_info['scp_codes']}")


np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

# Show feature vector
feature_vector = features[0][0]  
print(f"\nFeature Vector (length {len(feature_vector)}):")
print(feature_vector)


# In[18]:


from utils import utils
from models.fastai_model import fastai_model
import pickle
import numpy as np
from scipy.special import softmax
import torch

# Model parameters
sampling_frequency = 100
datafolder = '../data/ptbxl/'
task = 'superdiagnosticNORMandMI'
outputfolder = '../output/'
modelname = 'fastai_resnet1d_wang'
num_classes = 2  
input_shape = [1000, 12]
experiment = 'exp1.1.1'  
mpath = f'../output/{experiment}/models/{modelname}/'
pretrainedfolder = mpath

# Load the MI and NORM data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
data, labels, Y, _ = utils.select_data(
    data, labels, task, min_samples=0, outputfolder=outputfolder
)

i = 200  # ECG index
ecg_sample = data[i]
label_info = labels.iloc[i]




# Load standardizer for preprocessing and preprocess ECG
standard_scaler = pickle.load(
    open(f'../output/{experiment}/data/standard_scaler.pkl', "rb")
)
ecg_sample_batch = np.expand_dims(ecg_sample, axis=0) 
ecg_sample_standardized = utils.apply_standardizer(
    ecg_sample_batch, standard_scaler
)


# Load model
model = fastai_model(
    modelname,
    num_classes,
    sampling_frequency,
    mpath,
    input_shape=input_shape,
    pretrainedfolder=pretrainedfolder,
    n_classes_pretrained=num_classes,
    pretrained=True,
    epochs_finetuning=0,
)


# Load model using dummy data to get the underlying model to inspect arcitecture and exract feature
X = [ecg_sample_standardized[0]]
y_dummy = [0]  
learn = model._get_learner(X, y_dummy, X, y_dummy)
learn.load(model.name)
pytorch_model = learn.model
pytorch_model.eval()

# Attach hook
features = []
def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())

# Inspect model
print(pytorch_model)
print("\nModel Head structure")
print(pytorch_model[-1])

# Attach the hook to the head layer
hook = pytorch_model[-1][-3].register_forward_hook(hook_fn)


ecg_sample_tensor = torch.tensor(ecg_sample_standardized, dtype=torch.float32)
ecg_sample_tensor = ecg_sample_tensor.permute(0, 2, 1)
ecg_sample_tensor = ecg_sample_tensor.to(learn.data.device)

# Run the model on the ecg
with torch.no_grad():
    outputs = pytorch_model(ecg_sample_tensor)

hook.remove()


# In[19]:


import numpy as np

# Print label info
print(f"ECG index: {i}")
print(f"Patient ID: {label_info['patient_id']}")
print(f"Filename: {label_info['filename_lr']}")
print(f"Recording Date: {label_info['recording_date']}")
print(f"scp codes: {label_info['scp_codes']}")


np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

# Show feature vector
feature_vector = features[0][0]  
print(f"\nFeature Vector (length {len(feature_vector)}):")
print(feature_vector)


# In[20]:


from utils import utils
from models.fastai_model import fastai_model
import pickle
import numpy as np
from scipy.special import softmax
import torch

# Model parameters
sampling_frequency = 100
datafolder = '../data/ptbxl/'
task = 'superdiagnosticNORMandMI'
outputfolder = '../output/'
modelname = 'fastai_inception1d'
num_classes = 2  
input_shape = [1000, 12]
experiment = 'exp1.1.1'  
mpath = f'../output/{experiment}/models/{modelname}/'
pretrainedfolder = mpath

# Load the MI and NORM data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
data, labels, Y, _ = utils.select_data(
    data, labels, task, min_samples=0, outputfolder=outputfolder
)

i = 200  # ECG index
ecg_sample = data[i]
label_info = labels.iloc[i]




# Load standardizer for preprocessing and preprocess ECG
standard_scaler = pickle.load(
    open(f'../output/{experiment}/data/standard_scaler.pkl', "rb")
)
ecg_sample_batch = np.expand_dims(ecg_sample, axis=0) 
ecg_sample_standardized = utils.apply_standardizer(
    ecg_sample_batch, standard_scaler
)


# Load model
model = fastai_model(
    modelname,
    num_classes,
    sampling_frequency,
    mpath,
    input_shape=input_shape,
    pretrainedfolder=pretrainedfolder,
    n_classes_pretrained=num_classes,
    pretrained=True,
    epochs_finetuning=0,
)


# Load model using dummy data to get the underlying model to inspect arcitecture and exract feature
X = [ecg_sample_standardized[0]]
y_dummy = [0]  
learn = model._get_learner(X, y_dummy, X, y_dummy)
learn.load(model.name)
pytorch_model = learn.model
pytorch_model.eval()

# Attach hook
features = []
def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())

# Inspect model
print(pytorch_model)
print("\nModel Head structure")
print(pytorch_model.layers[1])

# Attach the hook to the head layer
hook = pytorch_model.layers[1][-3].register_forward_hook(hook_fn)


ecg_sample_tensor = torch.tensor(ecg_sample_standardized, dtype=torch.float32)
ecg_sample_tensor = ecg_sample_tensor.permute(0, 2, 1)
ecg_sample_tensor = ecg_sample_tensor.to(learn.data.device)

# Run the model on the ecg
with torch.no_grad():
    outputs = pytorch_model(ecg_sample_tensor)

hook.remove()


# In[21]:


import numpy as np

# Print label info
print(f"ECG index: {i}")
print(f"Patient ID: {label_info['patient_id']}")
print(f"Filename: {label_info['filename_lr']}")
print(f"Recording Date: {label_info['recording_date']}")
print(f"scp codes: {label_info['scp_codes']}")


np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

# Show feature vector
feature_vector = features[0][0]  
print(f"\nFeature Vector (length {len(feature_vector)}):")
print(feature_vector)


# In[ ]:




