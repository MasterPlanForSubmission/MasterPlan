# -*- encoding: utf-8 -*-
'''
@File : utils.py
@Describe : Some function for the project
@Time : 2022/07/19 14:48:47
'''

import os
import argparse
import torch
import pickle
import shutil
import sys
sys.path.append("..")
from structure.config import RANDOM_SEED, DISK_LIST_NUM
import random
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def set_seed(random_seed=RANDOM_SEED):
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def test_seed(random_seed=RANDOM_SEED):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def remove_test_seed():
    torch.manual_seed(torch.initial_seed())
    torch.cuda.manual_seed(torch.initial_seed())
    
def thresh_check(value):
    value = float(value)
    try:
        if value <= 0 or value >= 1:
            raise argparse.ArgumentTypeError('Action threshold must be within (0, 1).')
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value provided.")
    return value

def arch_check(value):
    value = float(value)
    try:
        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError('Archive task rate must be within (0, 1).')
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value provided.")
    return value
        
def normalize(state):
    return (state - np.min(state)) / (np.max(state) - np.min(state) + 1e-8)

def print_begin(file_name, max_poweron_num):
    with open(f'{file_name}', 'a+') as f:
        for i in range(max_poweron_num):
            f.write(f"AS 0 DG {i} is power-on at timestamp 60.0!" + '\n')
        
def print_reset(file_name, results, finish_task_num, task_num, episode_num, run_time, reward, avg_on_time=False):
    with open(f'{file_name}', 'a+') as f:
        if not avg_on_time:
            f.write(f'Time: {run_time:.2f}s, {task_num} tasks, finish {finish_task_num}, latency: {results[0]:.2f}s, throughput: {results[-1][0]:.2f}Gbps, archive: {results[-1][1]:.2f}Gbps, restore: {results[-1][2]:.2f}Gbps' + '\n')
            f.write(f'Finish episode {episode_num}, reward: {reward:.2f}' + '\n')
            f.write('\n')
        else:
            f.write(f'Time: {run_time:.2f}s, {task_num} tasks, finish {finish_task_num}, avg on time: {avg_on_time:.2f}s, latency: {results[0]:.2f}s, throughput: {results[-1][0]:.2f}Gbps, archive: {results[-1][1]:.2f}Gbps, restore: {results[-1][2]:.2f}Gbps' + '\n')
            f.write(f'Finish episode {episode_num}, reward: {reward:.2f}' + '\n')
            f.write('\n')
            
def print_eval(file_name, results, finish_task_num, task_num, eval_num, run_time, reward, avg_on_time=False):
    with open(f'{file_name}', 'a+') as f:
        if not avg_on_time:
            f.write(f'Time: {run_time:.2f}s, {task_num} tasks, finish {finish_task_num}, latency: {results[0]:.2f}s, throughput: {results[-1][0]:.2f}Gbps, archive: {results[-1][1]:.2f}Gbps, restore: {results[-1][2]:.2f}Gbps' + '\n')
            f.write(f'Finish evaluate {eval_num}, reward: {reward:.2f}' + '\n')
            f.write('\n')
        else:
            f.write(f'Time: {run_time:.2f}s, {task_num} tasks, finish {finish_task_num}, avg on time: {avg_on_time:.2f}s, latency: {results[0]:.2f}s, throughput: {results[-1][0]:.2f}Gbps, archive: {results[-1][1]:.2f}Gbps, restore: {results[-1][2]:.2f}Gbps' + '\n')
            f.write(f'Finish evaluate {eval_num}, reward: {reward:.2f}' + '\n')
            f.write('\n')

def print_test(file_name, results, finish_task_num, task_num, eval_num, run_time, reward, latencies):
    with open(f'{file_name}', 'a+') as f:
        f.write(f'Time: {run_time:.2f}s, {task_num} tasks, finish {finish_task_num}, latency: {results[0]:.2f}s, throughput: {results[-1][0]:.2f}Gbps, archive: {results[-1][1]:.2f}Gbps, restore: {results[-1][2]:.2f}Gbps' + '\n')
        print(f'Tail 99% latency is: {latencies[0]:.2f}s, tail latency is: {latencies[1]:.2f}s')
        f.write(f'Finish test {eval_num}, reward: {reward:.2f}' + '\n')
        f.write('\n')
            
def init_folder(log_folder, checkpoint_folder, trainres_folder, test_flag):
    if test_flag == 0:
        if os.path.exists(log_folder):
            shutil.rmtree(log_folder)
            os.makedirs(log_folder)
        else:
            os.makedirs(log_folder)
        if os.path.exists(trainres_folder):
            shutil.rmtree(trainres_folder) 
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(trainres_folder):
        os.makedirs(trainres_folder)

# generate normal distribution data point
def normal_distribution(mu, sigma):
    return abs(np.random.normal(mu, sigma, 1)[0])

def generate_bad_record(args, archive_pod):
    bad_records = [] # all bad disk info
    for i in range(args.bad_disk_num):
        AS_idx, AS_chosen = random.choice(list(enumerate(archive_pod.ASs)))
        dg_idx, dg_chosen = random.choice(list(enumerate(AS_chosen.dgs)))
        disk_idx, disk_chosen = random.choice(list(enumerate(dg_chosen.smr_disks)))
        bad_time = random.random() * args.max_time / 20 
        bad_records.append([AS_idx, dg_idx, disk_idx, bad_time])
    bad_records = sorted(bad_records, key=lambda x:x[-1])
    with open('./data/bad_record.pkl', 'wb') as f:
        pickle.dump(bad_records, f)
    return bad_records

def poisson_distribution(lam):
    return np.random.poisson(lam=lam, size=1)


def generate_disk_list(hw_config):
    avg_disk_num = int(hw_config["save_num"] / hw_config["rack_num"])
    multi_num = hw_config["save_num"] % hw_config["rack_num"]  
    disk_in_rack = hw_config["smr_num"] // hw_config["rack_num"] 
    assert disk_in_rack % hw_config["jbod_num"] == 0
    disk_in_jbod = int(disk_in_rack / hw_config["jbod_num"])
    total_archive_id = []
    for _ in range(DISK_LIST_NUM):
        archive_id = []
        multi_rack_id = np.random.choice(hw_config["rack_num"], multi_num, replace=False)
        for j in range(hw_config["rack_num"]):
            if isinstance(multi_rack_id, np.ndarray) and j in multi_rack_id:
                disk_num = avg_disk_num + 1
            else:
                disk_num = avg_disk_num
            disk_num_in_jbod = disk_num // hw_config["jbod_num"] 
            if disk_num_in_jbod == 0: 
                jbod_choose = np.random.choice(hw_config["jbod_num"], disk_num, replace=False)
                jbod_choose.sort()
                for k in jbod_choose:
                    archive_id.append(k*disk_in_jbod + j*disk_in_jbod*hw_config["jbod_num"] + random.randint(0, disk_in_jbod-1))
            else: 
                multi_jbod_num = disk_num % hw_config["jbod_num"] 
                jbod_choose_multi = np.random.choice(hw_config["jbod_num"], multi_jbod_num, replace=False)
                for k in range(hw_config["jbod_num"]):
                    if isinstance(jbod_choose_multi, np.ndarray) and k in jbod_choose_multi:
                        chosen = np.random.choice(disk_in_jbod, disk_num_in_jbod+1, replace=False) + k*disk_in_jbod + j*disk_in_jbod*hw_config["jbod_num"] 
                        archive_id.extend(chosen)
                    else:
                        chosen = np.random.choice(disk_in_jbod,disk_num_in_jbod, replace=False) + k*disk_in_jbod + j*disk_in_jbod*hw_config["jbod_num"] 
                        archive_id.extend(chosen)
        assert len(archive_id) == hw_config["save_num"] 
        archive_id.sort()
        parity_id = np.random.choice(archive_id, hw_config["parity_num"], replace=False).tolist()
        total_archive_id.append([archive_id, parity_id])
    np.save('./data/disk_list.npy', np.array(total_archive_id, dtype=object))
    