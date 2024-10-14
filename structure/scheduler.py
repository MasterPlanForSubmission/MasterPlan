# -*- encoding: utf-8 -*-
'''
@File : scheduler.py
@Describe :
@Time : 2023/03/06 11:44:12
'''

import sys
import numpy as np
from structure.config import MAX_ARCHIVE, MAX_RESTORE, MAX_SCRUB, MAX_REPAIR, CLASS_IDX, PRIORITY_NAMES, ZONE_SIZE, REPAIR_WEIGHT, LIST_WEIGHT
from structure.disk import DiskGroup
from util.pq import PriorityQueue
class Sche_IO:
    '''
        task_num: allocated task nums
        finish_tasks: finished tasks every interval
    '''
    def __init__(self, archive_server, dg_sche_io: list['DG_Sche_IO']):
        self.time = 0.0
        self.archive_server = archive_server
        self.task_num = 0
        self.finish_tasks = 0
        self.dg_sche_io = dg_sche_io
        self.throughput = 0 
        self.finishes = 0 
        
    def reset(self):
        self.throughput, self.finishes, self.finish_tasks, self.task_num = 0, 0, 0, 0
        self.time = 0.0
        for i in range(len(self.dg_sche_io)):
            self.dg_sche_io[i].reset()
            
    def update_exec_task(self, cannot_exec):
        if len(self.archive_server.poweron_dg) == 0:
            return []
        finished = []
        for dg_id in self.archive_server.poweron_dg:
            flag = True
            finish_task, total_throughput = self.dg_sche_io[int(dg_id)].update_exec_task(round(self.time, 1))
            self.throughput += total_throughput
            self.task_num -= len(finish_task)
            finished.extend(finish_task)
            if [self.archive_server.id, dg_id] in cannot_exec:
                flag = False
            self.dg_sche_io[int(dg_id)].exec_IO(flag)
            self.finish_tasks += len(finished)
            self.finishes += len(finished)
        return finished
        
        
class DG_Sche_IO:
    '''
        AS_DG_NUM dg_scheduler for each AS
        pq_0: scrub priority queue
        pq_1: restore low priority queue
        pq_2: restore medium priority queue
        pq_3: restore urgent priority queue
        pq_4: archive priority queue
        pq_5: repair priority queue
        Qk: record the labels of the most recent queued tasks in the corresponding task category
        Q: the digital labels of the last dequeued tasks for all categories
        qt_size: the total amount of tasks for each category
    '''
    
    def __init__(self, dg: 'DiskGroup', pq_num: int):
        self.dg = dg
        self.pq = [PriorityQueue() for _ in range(pq_num)]
        self.Qk = [0.0] * pq_num
        self.Q = 0
        self.exec_tasks = {"archive": [], "restore": [], "scrub": [], "repair": []}
        self.qt_size = [0.0] * pq_num
        self.hang_flag = False
        
    def reset(self):
        self.pq = [PriorityQueue() for _ in range(len(self.pq))]
        self.Qk = [0.0] * len(self.Qk)
        self.Q = 0
        self.exec_tasks = {"archive": [], "restore": [], "scrub": [], "repair": []}
        self.qt_size = [0.0] * len(self.qt_size)
        self.hang_flag = False
        
    def set_task(self, task):
        task_type, index = task.task_name, 0
        if task.task_name == "restore":
            task_type += '_' + PRIORITY_NAMES[task.priority]
        index = CLASS_IDX[task_type]
        if task.task_name != "repair":
            task.weight = max(self.Qk[index], self.Q) + task.task_size / LIST_WEIGHT[index]
        else:
            task.weight = max(self.Qk[index], self.Q) + task.task_size / REPAIR_WEIGHT[task.priority]
        self.pq[index].put(task)
        self.qt_size[index] += task.task_size
        self.Qk[index] = task.weight
        
    def exist_task(self):
        return len(self.exec_tasks["archive"]) > 0 or len(self.exec_tasks["restore"]) > 0 or len(self.pq[4].queue)
    
    def exec_IO(self, not_archive_flag):
        if not not_archive_flag:
            while len(self.exec_tasks["archive"]) < MAX_ARCHIVE and len(self.pq[4].queue) > 0:
                t = self.pq[4].queue[0]
                disks_id, parity_id = self.dg.choose_disk(t.task_id)
                t.location.append(disks_id)
                t.parity_location = parity_id
                for d_id in t.location[-1]:
                    self.dg.smr_disks[d_id].fifo_exec.append([t, ZONE_SIZE])
                self.pq[4].get()
                self.Q = t.weight
                self.exec_tasks["archive"].append(t)
            return
        
        cannot_choose = [] 
        # get task from task_queue
        while len(self.exec_tasks["archive"]) < MAX_ARCHIVE or len(self.exec_tasks["restore"]) < MAX_RESTORE:
            head_arr, min_index, flag = [], -1, False 
            chose_id = list(range(len(self.pq)))
            for i in chose_id:
                if i in cannot_choose or len(self.pq[i].queue) == 0:
                    head_arr.append(sys.maxsize)
                    continue
                if len(self.pq[i].queue) > 0:
                    head_arr.append(self.pq[i].queue[0].weight)
                    flag = True
            if not flag:
                return
            min_index = np.argmin(head_arr)
            t = self.pq[min_index].queue[0]
            min_flag = False # Indicates whether the extracted task can be executed, if not, let other tasks continue
            if not self.hang_flag and min_index == 0 and len(self.exec_tasks["scrub"]) < MAX_SCRUB:
                min_flag = True
                for d_id in t.location[-1]:
                    self.dg.smr_disks[d_id].fifo_exec.append([t, ZONE_SIZE])
                self.pq[min_index].get()
                self.Q = t.weight
                self.exec_tasks["scrub"].append(t)
            elif 1 <= min_index <= 3 and len(self.exec_tasks["restore"]) < MAX_RESTORE:
                min_flag = True
                for disk_id in t.replica_size.keys():
                    self.dg.smr_disks[int(disk_id)].fifo_exec.append([t, t.replica_size[disk_id]])
                self.pq[min_index].get()
                self.Q = t.weight
                self.exec_tasks["restore"].append(t)
            elif min_index == 4 and len(self.exec_tasks["archive"]) < MAX_ARCHIVE:
                min_flag = True
                disks_id, parity_id = self.dg.choose_disk(t.task_id)
                t.location.append(disks_id)
                t.parity_location = parity_id
                for d_id in t.location[-1]:
                    self.dg.smr_disks[d_id].fifo_exec.append([t, ZONE_SIZE])
                self.pq[min_index].get()
                self.Q = t.weight
                self.exec_tasks["archive"].append(t)
            elif not self.hang_flag and min_index == 5 and len(self.exec_tasks["repair"]) < MAX_REPAIR:
                min_flag = True
                for d_id in t.location[-1]:
                    self.dg.smr_disks[d_id].fifo_exec.append([t, ZONE_SIZE])
                self.pq[min_index].get()
                self.Q = t.weight
                self.exec_tasks["repair"].append(t)
            if self.pq[0].qsize()+self.pq[1].qsize()+self.pq[2].qsize()+self.pq[3].qsize()+self.pq[4].qsize()+self.pq[5].qsize() == 0:
                break
            if not min_flag:
                if min_index == 1 or min_index == 2 or min_index == 3:
                    for k in range(1, 4):
                        cannot_choose.append(k)
                else:
                    cannot_choose.append(min_index)
                    
                    
    def update_exec_task(self, time):
        finished = []
        total_throughput = 0
        if len(self.exec_tasks["archive"]) > 0 or len(self.exec_tasks["restore"]) > 0 or len(self.exec_tasks["repair"]) > 0:
            for i, disk in enumerate(self.dg.smr_disks):
                if not disk.flag:
                    continue
                finish_task, archive_size, throughput_size = disk.do_io(time)
                self.dg.cur_size += archive_size
                total_throughput += throughput_size
                finished.extend(finish_task)
            self.update_finish(finished, time)
        return finished, total_throughput
    
    def update_finish(self, finished, time):
        for t in finished:
            assert t.stripe_num == t.task_size
            self.dg.allocated_tasks -= 1
            task_type, index = t.task_name, 0
            if t.task_name == "restore":
                task_type += '_' + PRIORITY_NAMES[t.priority]
            elif t.task_name == "repair":
                disk_id = t.location[2]
                self.dg.finish_chunk[disk_id] += 1
                if self.dg.finish_chunk[disk_id] == self.dg.max_chunk[disk_id]:
                    self.dg.smr_disks[disk_id].flag = 1
                    self.dg.bad_disks.remove(disk_id)
                    del self.dg.finish_chunk[disk_id]
                    del self.dg.max_chunk[disk_id]
            elif t.task_name == "scrub":
                self.dg.chunk_queue[t.location[2]][0] = time
                self.dg.chunk_queue.sort(key=lambda x: x[0])
            self.exec_tasks[t.task_name].remove(t)
            index = CLASS_IDX[task_type]
            self.qt_size[index] -= t.task_size
            t.done = True
            t.finish_timestamp = time
            # print(f"Finish {t.task_name} task with id: {t.task_id}, at timestamp {time:.3f}")
            
    def remove_off_tasks(self):
        for type in ["scrub", "repair"]:
            for t in self.exec_tasks[type]:
                self.change_task_state(t)
                
    def change_task_state(self, t):
        t.stripe_num = 0 # need redo
        for d_id in t.location[-1]:
            flag = False
            for w_t in self.dg.smr_disks[int(d_id)].fifo_exec:
                if w_t[0] == t:
                    w_t[1] = ZONE_SIZE
                    flag = True
                    break
                if not flag:
                    self.dg.smr_disks[int(d_id)].fifo_exec.append([t, ZONE_SIZE])
