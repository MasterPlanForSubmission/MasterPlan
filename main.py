# -*- encoding: utf-8 -*-
'''
@File : main.py
@Describe : main function for MasterPlan
@Time : 2023/03/06 11:00:59
'''

import time
import json
import argparse
import pickle
from algorithm.starter import Starter
from algorithm.PPO import PPOAgent
from structure.config import PQ_NUM, URGENT_POSS, COMMON_POSS
from util.utils import generate_disk_list, set_seed, init_folder, generate_bad_record, thresh_check, arch_check
from util.metrics import Metrics
from structure.timer import Timer
from structure.task import generate_task
from structure.scheduler import Sche_IO, DG_Sche_IO
from structure.archive_server import Archive_Server
from structure.archive_pod import Archive_Pod
from structure.disk import DiskGroup

def get_args():
    parser = argparse.ArgumentParser(description='Training parameters for the RL agent')
    parser.add_argument('--mode', type=str, help='Mode for train or test', default="train", required=False, choices=["train", "test"])
    parser.add_argument('--hardware', type=str, help='Hardware parameter set', default="1", choices=["1", "2", "3"])
    parser.add_argument('--max_time', type=int, help='Max training episode time', default=3000)
    parser.add_argument('--workload_rate', type=float, help='Input workload rate', default=4)
    parser.add_argument('--archive_rate', type=arch_check, help='Input archive task rate', default=0.5)
    parser.add_argument('--urgent_flag', type=bool, help='Most of restore tasks are urgent or not', default=False)
    parser.add_argument('--chunk_num', type=int, help='Already saved chunk num for each disk group', default=100)
    parser.add_argument('--bad_disk_num', type=int, help='Bad disk number', default=0)
    parser.add_argument('--scrub_flag', type=bool, help='Generate scrub task or not', default=False)
    parser.add_argument('--act_threshold', type=thresh_check, help='Threshold of action', default=0.85)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--pos_weight', type=float, help='Weight of positive reward', default=0.3)
    parser.add_argument('--update_timestep', type=int, help='Timestep interval of updating the model', default=1024)
    parser.add_argument('--save_timestep', type=int, help='Timestep interval of saving the model', default=100000)
    parser.add_argument('--eval_episode', type=int, help='Episode interval of evaluating the model', default=1000)
    parser.add_argument('--model_name', type=str, help='Name of the test model, best or saving timesteps', default='best')
    args = parser.parse_args() 
    if args.urgent_flag:
        args.prio_poss = URGENT_POSS
    else:
        args.prio_poss = COMMON_POSS
    return args

if __name__ == '__main__':
    args = get_args()
    with open('./data/hw_setup.json', 'r') as file:
        data = json.load(file)
    hw_config = data[args.hardware]
    start_time = time.perf_counter()
    print('Start simulating!')
    set_seed()
    generate_disk_list(hw_config)
    disk_groups = [[] for _ in range(hw_config["as_dg_num"])]
    for i in range(hw_config["as_dg_num"]):
        dg = DiskGroup(id=i, chunk_num=args.chunk_num, hw_config=hw_config)
        dg.generate_chunks(0)
        disk_groups[0].append(dg)
        
    archive_server_1 = Archive_Server(id=0, dgs=disk_groups[0], hw_config=hw_config)
    archive_pod = Archive_Pod(as_list=[archive_server_1])
    archive_pod.record_write()
    tasks = generate_task(args, hw_config, archive_pod)
    # with open('./data/task_queue.pkl', 'rb') as f:
    #     tasks = pickle.load(f)
    bad_records = generate_bad_record(args, archive_pod)
    
    dg_sche_io_1 = [DG_Sche_IO(dg=disk_groups[0][i], pq_num=PQ_NUM) for i in range(hw_config["as_dg_num"])]
    sche_IO_1 = Sche_IO(archive_server=archive_server_1, dg_sche_io=dg_sche_io_1)
    sche_IO = [sche_IO_1]
    metrics = Metrics()
    folder_name = f"{args.max_time}_{args.workload_rate}_{args.archive_rate}"
    log_folder = f"./print_log/{folder_name}"
    checkpoint_folder = f"./checkpoint/{folder_name}"
    trainres_folder = f"./train/{folder_name}"
    ppoagent = PPOAgent(args=args, as_dg_num=hw_config["as_dg_num"], dgs=disk_groups[0])
    
    timer = Timer(args=args, hw_config=hw_config, archive_pod=archive_pod, archive_server=archive_server_1, metrics=metrics, sche_io=sche_IO_1, tasks=tasks, bad_records=bad_records)
    
    starter = Starter(args=args, timer=timer, archive_pod=archive_pod, agent=ppoagent, sche_io=sche_IO_1, metrics=metrics, folder_name=folder_name)

    if args.mode == 'train':
        init_folder(log_folder, checkpoint_folder, trainres_folder, 0)
        starter.run()
    else:
        init_folder(log_folder, checkpoint_folder, trainres_folder, 1)
        starter.test()
    