import librosa
import numpy as np
import os
import sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
import time

def wav_rename(wavpath, speaker):
    name_path = os.path.join(wavpath, speaker)
    wavs = os.listdir(name_path)
    for w in tqdm(wavs):
        old_name = name_path + "/" + w
        new_name = name_path + "/" + speaker + w
        os.rename(old_name,new_name)


def get_spk_world_feats(train_wavpath, test_wavpath, speaker, mc_dir_train, mc_dir_test, sample_rate=16000):
    train_speaker_path = os.path.join(train_wavpath, speaker)
    test_speaker_path = os.path.join(test_wavpath, speaker)

    train_paths = os.listdir(train_speaker_path)
    test_paths = os.listdir(test_speaker_path)

    f0s = []
    coded_sps = []
    for wav_file in train_paths:
        wav_file = os.path.join(train_speaker_path, wav_file)
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)
    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
    np.savez(os.path.join(mc_dir_train, speaker+'_stats.npz'), 
            log_f0s_mean=log_f0s_mean,
            log_f0s_std=log_f0s_std,
            coded_sps_mean=coded_sps_mean,
            coded_sps_std=coded_sps_std)
    
    for wav_file in tqdm(train_paths):
        wav_file = os.path.join(train_speaker_path, wav_file)
        wav_nam = os.path.basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(os.path.join(mc_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    
    for wav_file in tqdm(test_paths):
        wav_file = os.path.join(test_speaker_path, wav_file)
        wav_nam = os.path.basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(os.path.join(mc_dir_test, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    return f"{speaker} Done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    sample_rate_default = 16000
    train_wavpath_default = "./data/vcc2016_training"
    test_wavpath_default = "./data/evaluation_all"
    mc_dir_train_default = './data/mc/train'
    mc_dir_test_default = './data/mc/test'

    parser.add_argument("--sample_rate", type = int, default = 16000, help = "Sample rate.")
    parser.add_argument("--train_wavpath", type = str, default = train_wavpath_default, help = "The directory to store the training data.")
    parser.add_argument("--test_wavpath", type = str, default = test_wavpath_default, help = "The directory to store the test data.")
    parser.add_argument("--mc_dir_train", type = str, default = mc_dir_train_default, help = "The directory to store the training features.")
    parser.add_argument("--mc_dir_test", type = str, default = mc_dir_test_default, help = "The directory to store the testing features.")
    parser.add_argument("--num_workers", type = int, default = None, help = "The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    train_wavpath = argv.train_wavpath
    test_wavpath = argv.test_wavpath
    mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()


    # WE only use 10 speakers listed below for this experiment.
    speaker_used = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

    print("Rename the wav")
    for spk in speaker_used:
        wav_rename(train_wavpath, spk)
        wav_rename(test_wavpath, spk)

    ## Next we are to extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds). 
    # Make dirs to contain the MCEPs
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = len(speaker_used) #cpu_count()
    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    # work_dir = target_wavpath
    # spk_folders = os.listdir(work_dir)
    # print("processing {} speaker folders".format(len(spk_folders)))
    # print(spk_folders)
    start = time.time()
    futures = []
    for spk in speaker_used:
        futures.append(executor.submit(partial(get_spk_world_feats, train_wavpath, test_wavpath, spk, mc_dir_train, mc_dir_test, sample_rate)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)

    end = time.time()
    all_time = end - start
    print("preprocess time : {all_time}")
    sys.exit(0)



