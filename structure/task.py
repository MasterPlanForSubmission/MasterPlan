# -*- encoding: utf-8 -*-
'''
@File : task.py
@Describe : Generate new task and task class
@Time : 2023/02/06 21:57:08
'''

from structure.config import ZONE_SIZE, PRIORITY, REPAIR_TIME, TIMER_INTERVAL
from structure.archive_pod import Archive_Pod
import random
import numpy as np
import pickle
class Task_Generator:
    '''
        generate all possible tasks(with time stamp) which are saved in a
        task queue,
        and send the task queue to the scheduler.
        write_finish_timestamp
    '''
    def __init__(self, archive_pod: Archive_Pod, hw_config):
        self.archive_pod = archive_pod
        self.hw_config = hw_config
        
    def generate_poisson_task(self, args, id, cur_time):
        cur_time = round(cur_time, 2)
        random_val = random.random() 
        
        # archive
        if random_val <= args.archive_rate:
            priority = PRIORITY[1]
            task = Task(id, cur_time, "archive", self.hw_config["chunk_size"], priority=priority, location=None, replica_size=None)
            id += 1
            
        # restore
        else:
            replica_size = {}
            priority = np.random.choice(PRIORITY, p=args.prio_poss)
            random_restore_size = int(random.uniform(1,
            self.hw_config["restore_num"] * ZONE_SIZE))
            arch = random.choice(self.archive_pod.write_info[:self.hw_config["as_dg_num"] * args.chunk_num])
            last_scrub_time, as_id, dg_id, restore_id, parity_id = arch
            if self.hw_config["restore_num"] * ZONE_SIZE < random_restore_size:
                random_restore_size = self.hw_config["restore_num"] * ZONE_SIZE
            disk_num = self.hw_config["restore_num"] if random_restore_size >= self.hw_config["restore_num"] else random_restore_size
            for i, d_id in enumerate(restore_id[:disk_num]):
                extra_num = 1 if i < random_restore_size % self.hw_config["restore_num"] else 0
                replica_size[d_id] = np.floor(random_restore_size / self.hw_config["restore_num"]) + extra_num
            task = Task(id, cur_time, "restore", random_restore_size, priority=priority, location=[as_id, dg_id, restore_id],replica_size=replica_size)
            id += 1
            
        return task
    
    
class Task:
    '''
        task_id: The uuid of a task
        time_stamp: The time stamp when creating the task
        task_name: The name of the task, "restore" or "archive" for user,
        "repair" or "scrub" for system
        done: Whether the task is completed
        task_size: The size of write/read tasks, e.g., the task_size of "archive
        100MB file at location A" is 100 MB
        location: The location for restore task, [as id, dg id, disks id], [int, int];
        replica_size: {disk_id: io_size}, for restore task
        priority: Time the task need to be finished
    '''
    def __init__(self, task_id, time_stamp, name, task_size, priority, location, replica_size):
        self.task_id = task_id
        self.time_stamp = time_stamp
        self.task_name = name
        self.done = False
        self.task_size = task_size
        self.priority = priority
        if self.task_name == "archive":
            self.parity_location = []
        self.location = location
        self.replica_size = replica_size
        self.start_timestamp = None
        self.finish_timestamp = None
        self.stripe_num = 0
        self.allocated = False
        self.weight = 0
            
    def __lt__(self, others):
        return self.weight < others.weight
    
    def get_time_stamp(self):
        return self.time_stamp
    
def generate_task(args, hw_config, archive_pod):
    if args.workload_rate == 0:
        return []
    not_poisson_time, not_interval_time = 60, 1 / args.workload_rate
    task_queue = []
    task_num, rand_n = [], []
    task_generator = Task_Generator(archive_pod=archive_pod, hw_config=hw_config)
    if TIMER_INTERVAL > 1 / args.workload_rate:
        cur_time = not_poisson_time
        multi_num = int(args.workload_rate % (1 / TIMER_INTERVAL))
        for i in np.arange(0, args.max_time - not_poisson_time, TIMER_INTERVAL):
            avg_num = int(args.workload_rate / (1 / TIMER_INTERVAL))
            if abs(i % 1) < 1e-5: 
                rand_n = np.random.choice(list(range(int(1 / TIMER_INTERVAL))), multi_num, replace=False)
                idx = round((i * 10) % 10)
                if idx in rand_n:
                    avg_num += 1
                task_num.append(avg_num)
            print(f"Total task num: {np.sum(task_num)}")
            for k in range(len(task_num)):
                for i in range(task_num[k]):
                    generated_t = task_generator.generate_poisson_task(args, len(task_queue), cur_time=cur_time)
                    task_queue.append(generated_t)
                    print(f'Add {generated_t.task_name} task at timestamp {cur_time:.2f}, id: {generated_t.task_id}, size: {generated_t.task_size:.1f}, finish before {generated_t.priority + cur_time:.2f}')
                cur_time += TIMER_INTERVAL
    else:
        cur_time = not_poisson_time
        while abs(cur_time - args.max_time) > 1e-5:
            if abs(cur_time - not_poisson_time) < 1e-5 or cur_time > not_poisson_time:
                generated_t = task_generator.generate_poisson_task(args, len(task_queue), cur_time=cur_time)
                task_queue.append(generated_t)
                print(f'Add {generated_t.task_name} task at timestamp {cur_time:.2f}, id: {generated_t.task_id}, size: {generated_t.task_size:.1f}, finish before {generated_t.priority + cur_time:.2f}')
                not_poisson_time += not_interval_time
            cur_time += TIMER_INTERVAL
        with open('./data/task_queue.pkl', 'wb') as f:
            pickle.dump(task_queue, f)
    return task_queue
    
# location scrub:[as_id, dg_id]; repair:[as_id, dg_id, disk_id]
def generate_background(id, hw_config, cur_time, type, location, bad_num=0):
    
    # scrub
    if type == "scrub":
        as_id, dg_id, chunk_id, disks_id = location 
        task = Task(id, cur_time, "scrub", hw_config["chunk_size"], priority=PRIORITY[2], location=[as_id, dg_id, chunk_id, disks_id], replica_size=None)
        return task
    
    # repair
    elif type == "repair":
        as_id, dg_id, disk_id, replica_disks = location
        remain_num = hw_config["parity_num"] - bad_num 
        assert remain_num >= 0
        if remain_num >= 3:
            priority = REPAIR_TIME
        elif remain_num == 2:
            priority = 2 * REPAIR_TIME / 3
        else:
            priority = REPAIR_TIME / 3
        task = Task(id, cur_time, "repair", ZONE_SIZE * hw_config["restore_num"] , location=[as_id, dg_id, disk_id, replica_disks], priority=priority,replica_size=None)
        return task
    