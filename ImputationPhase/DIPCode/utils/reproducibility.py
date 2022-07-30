import numpy as np
import os
import csv
from pathlib import Path
import random
from scipy.stats import bernoulli

def save_runs_shuffle(rep_dir, source, number_runs, number_observations):

    data = []

    for i in range(number_runs):
        run_shuffle = np.array(range(number_observations))
        np.random.shuffle(run_shuffle)
        data.append(run_shuffle)

    np_data = np.asarray(data)

    if not os.path.exists(rep_dir + source):
        os.makedirs(rep_dir + source)

    np.save(rep_dir + source + "/runs_shuffle.npy", np_data)

def get_run_shuffle(rep_dir, source, run):

    return np.load(rep_dir + source + "/runs_shuffle.npy", allow_pickle=True)[run]


def generate_missing_values_mcar(input_shape, p):
    
    # Bernoulli Distribution
    missing_val_perimage = np.ones((input_shape[0],input_shape[1],input_shape[2]))
    for i in range(input_shape[0]):
        M = bernoulli.rvs(p, size = (input_shape[1],input_shape[2])) ##Gera uma matriz de 1s que tem de ser invertida
        missing_val_perimage[i] = M
        
    return missing_val_perimage 


def get_missing_masks(rep_dir, source, shape, missing_rate):

    mask_neg = generate_missing_values_mcar([shape[0],shape[1],shape[2]], 1 - missing_rate)

    return mask_neg


def write_line_to_csv(dir_path, file, data_row):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file
    file_exists = Path(file_path).is_file()

    if file_exists:
        file_csv = open(file_path, 'a')
    else:
        file_csv = open(file_path, 'w')

    writer = csv.DictWriter(file_csv, delimiter=',', fieldnames=[*data_row], quoting=csv.QUOTE_NONE)

    if not file_exists:
        writer.writeheader()

    writer.writerow(data_row)

    file_csv.flush()
    file_csv.close()