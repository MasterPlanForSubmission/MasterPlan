# -*- encoding: utf-8 -*-
'''
    @File : disk.py
    @Describe : class of disk and diskgroup
    @Time : 2022/07/19 11:48:02
'''

from structure.config import OFFING_TIME, ONING_TIME, RW_SPEED, SMR_SIZE, DISK_LIST_NUM
import numpy as np
import random
import pickle

class Disk:
    '''
        total_size: The max size of disk, so the total_zone_num will be
        disk_size / ZONE_SIZE
        cur_size: The current size of the disk, cur_zone_num will be cur_size / ZONE_SIZE
        flag: Whether the disk is bad, 1 for good and 0 for bad
    '''
    def __init__(self, id: int, total_size: float, cur_size: float, flag: int):
        self.id = id
        self.total_size = total_size
        self.cur_size = cur_size
        self.flag = flag
        self.fifo_exec = []
        self.replica_queue = []
        
    def reset(self, size):
        self.cur_size = size
        self.flag = True
        self.fifo_exec = []
        self.replica_queue = []
        
    def get_cur_size(self):
        return self.cur_size
    
    # update task
    def do_io(self, time):
        if len(self.fifo_exec) == 0:
            return [], 0, 0
        io_num, init_io_num, archive_size = RW_SPEED, RW_SPEED, 0
        finished, removed = [], []
        exec_id = int(random.random() * len(self.fifo_exec))
        exec_list = [i for i in range(exec_id, len(self.fifo_exec))]
        exec_list.extend([i for i in range(exec_id)])
        for i in exec_list:
            w_t = self.fifo_exec[i] # w_t: [task, io_num]
            if w_t[0].stripe_num == 0:
                w_t[0].start_timestamp = time
            if w_t[1] < io_num or abs(w_t[1] - io_num) < 1e-4:
                removed.append(w_t)
                io_num -= w_t[1]
                w_t[0].stripe_num += w_t[1]
                if abs(w_t[0].stripe_num - w_t[0].task_size) < 1e-4 or w_t[0].stripe_num > w_t[0].task_size:
                    finished.append(w_t[0])
            else:
                w_t[0].stripe_num += io_num
                w_t[1] -= io_num
                io_num = 0
            if w_t[0].task_name == "archive":
                archive_size = init_io_num - io_num
                init_io_num = io_num
                self.cur_size += archive_size
            if io_num == 0:
                break
        for t in removed:
            self.fifo_exec.remove(t)
        throughput_size = RW_SPEED - io_num
        return finished, archive_size, throughput_size

class DiskGroup:
    '''
        num: The number of disks in a diskgroup. smr_size: The size of a smr disk.
        power_state: power state for dg, 4 possible state: 1."off"; 2."offing"; 3."oning"; 4."on".
        smr_disks: All disks in a diskgroup.
        poweroning_time_stamp: UNLOADED->LOADING
        poweron_time_stamp: LOADING->LOADED 
        poweroffing_time_stamp: LOADED->UNLOADING
        poweroff_time_stamp: UNLOADING->UNLOADED
        cur_size: The current size of the diskgroup, (todo: expressed in chunksize (5GB)).
        chunk_queue:  disk id: [task_id, [disks_id], last_time]
        on_time: power-on time stamp
        begin_chunk_id: begin chunk id for repair
        max_chunk_id: max chunk id for a bad disk
    '''

    def __init__(self, id: int, chunk_num, hw_config):
        self.id = id
        self.chunk_num = chunk_num
        self.hw_config = hw_config
        self.power_state = "off"
        self.smr_disks = [Disk(id=i, total_size=SMR_SIZE, cur_size=0, flag=1) for i in range(hw_config["smr_num"])]
        self.poweroning_time_stamp = 0.0
        self.poweron_time_stamp = 0.0
        self.poweroffing_time_stamp = 0.0
        self.poweroff_time_stamp = 0.0
        self.total_size = hw_config["smr_num"] * SMR_SIZE
        self.chunk_queue = []
        self.disk_list = np.load("./data/disk_list.npy", allow_pickle=True).tolist()
        self.allocated_tasks = 0
        self.on_time = []
        self.bad_disks = set()
        self.finish_chunk = dict() # {bad_disk_id: finish_chunk_num}
        self.max_chunk = dict() # {bad_disk_id: max_chunk_num}
        self.init_disk_size = [0 for _ in range(hw_config["smr_num"])] 
        self.cur_size = len(self.chunk_queue) * hw_config["chunk_size"]
        self.scrub_id = 0
        
    def reset(self):
        for i, disk in enumerate(self.smr_disks):
            disk.reset(self.init_disk_size[i])
        self.power_state = "off"
        self.poweroning_time_stamp, self.poweron_time_stamp, self.poweroffing_time_stamp, self.poweroff_time_stamp = 0.0, 0.0, 0.0, 0.0
        self.generate_chunks(0, True)
        self.cur_size = len(self.chunk_queue) * self.hw_config["chunk_size"]
        self.allocated_tasks = 0
        self.srcub_id = 0
        self.on_time = []
        self.bad_disks = set()
        self.finish_chunk, self.max_chunk = dict(), dict()
        return self.cur_size

    def get_cur_size(self):
        cur_size = 0
        for disk in self.smr_disks:
            cur_size += disk.get_cur_size()
        self.cur_size = cur_size
        return cur_size
    
    # archived chunks
    def generate_chunks(self, as_id, reset_flag=False):
        init_disk_size = [0.0 for _ in range(self.hw_config["smr_num"])]
        archive_times = []
        if reset_flag:
            with open(f"./chunk_queue/AS{as_id}_DG{self.id}_archive_time.pkl", "rb+") as f:
                archive_times = pickle.load(f)
        for i in range(self.chunk_num):
            disks = self.disk_list[i][0]
            parity_id = self.disk_list[i][1]
            restore_id = list(set(disks) - set(parity_id))
            if not reset_flag:
                archive_time = -random.random() * 24 * 3600 
                archive_times.append(archive_time)
            for j in disks:
                self.smr_disks[j].cur_size += self.hw_config["chunk_size"] / self.hw_config["save_num"]
                init_disk_size[j] += self.hw_config["chunk_size"] / self.hw_config["save_num"]
                self.smr_disks[j].replica_queue.append(len(self.chunk_queue))
                self.chunk_queue.append([archive_times[i], as_id, self.id, list(restore_id), list(parity_id)])
        self.chunk_queue.sort(key=lambda x: x[0])
        self.init_disk_size = init_disk_size
        # with open(f"./chunk_queue/AS{as_id}_DG{self.id}.pkl", "wb+") as q_file:
        #     pickle.dump(self.chunk_queue, q_file)
        if not reset_flag:
            with open(f"./chunk_queue/AS{as_id}_DG{self.id}_archive_time.pkl", "wb+") as f:
                pickle.dump(archive_times, f)
        
    def check_state(self, time, as_id, file_name):
        if self.power_state == "oning":
            past_on_time = time - self.poweroning_time_stamp
            if abs(past_on_time - ONING_TIME) < 1e-4 or past_on_time > ONING_TIME:
                self.power_state = "on"
                self.poweron_time_stamp = time
                with open(file_name, 'a+') as f:
                    f.write(f"AS {as_id} DG {self.id} is power-on at timestamp {time:.1f}!" + '\n')
            elif self.power_state == "offing":
                past_off_time = time - self.poweroffing_time_stamp
                if abs(past_off_time - OFFING_TIME) < 1e-4 or past_off_time > OFFING_TIME:
                    self.power_state = "off"
                    self.poweroff_time_stamp = time
                    with open(file_name, 'a+') as f:
                        f.write(f"AS {as_id} DG {self.id} is power-off at timestamp {time:.1f}!" + '\n')

    def choose_disk(self, task_id):
        # rand = random.randint(0, len(DISK_LIST)-1)
        disks_id = self.disk_list[task_id % DISK_LIST_NUM][0]
        parity_id = self.disk_list[task_id % DISK_LIST_NUM][1]
        return disks_id, parity_id
    
    