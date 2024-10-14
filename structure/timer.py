# -*- encoding: utf-8 -*-
'''
@File : timer.py
@Describe : simulator 
@Time : 2023/03/06 19:55:51
'''

from structure.config import TIMER_INTERVAL, POLICY_INTERVAL, BASELINE_POWERON_TIME, MAX_ARCHIVE, SCRUB_TIME, SCRUB_INTERVAL, RW_SPEED
from algorithm.algo_config import NEG_ON_TIME, NEG_OFF_TIME, POS_WEIGHT, NEG_WEIGHT
from structure.archive_pod import Archive_Pod
from structure.scheduler import Sche_IO
from structure.archive_server import Archive_Server
from structure.task import Task, generate_background
from util.metrics import Metrics
from util.utils import set_seed, normalize
import copy
import pickle
import random
import numpy as np


class Timer:
    def __init__(self, args, hw_config, archive_pod: Archive_Pod, archive_server: Archive_Server, metrics: Metrics, sche_io: Sche_IO, tasks: list[Task], bad_records, interval=POLICY_INTERVAL):
        self.args = args
        self.hw_config = hw_config
        self.time = 0.0
        self.archive_pod = archive_pod
        self.archive_server = archive_server
        self.metrics = metrics
        self.sche_io = sche_io
        self.gen_t = tasks
        self.interval = interval
        self.train_time = [500 * i for i in range(1, self.args.max_time // 500 + 1)]
        self.total_power = [] 
        self.cannot_exec = [] 
        self.oning_dg = [] 
        self.task_id = 0 # next insert task id
        self.archive_tasks = [] 
        self.bad_records = bad_records # bad disk location and time
        self.background_task = [] 
        
    def set_time(self, set_classes):
        for set_class in set_classes:
            if isinstance(set_class, list):
                for io in set_class:
                    io.time = self.time
            else:
                set_class.time = self.time
                
    def check_all(self, file_name):
        for dg_id in list(self.archive_server.withpower_dg):
            self.archive_server.dgs[dg_id].check_state(self.time, self.archive_server.id, file_name)
        self.archive_server.check_power_state()
        self.policy_mod = self.time % self.interval
        
    def check_clean_up(self):
        removed, policy = [], []
        for as_id, dg_id in self.cannot_exec:
            dg_id = int(dg_id)
            if not self.sche_io.dg_sche_io[dg_id].exist_task():
                removed.append([as_id, dg_id])
                policy.append(["off", as_id, [dg_id]])
                
        for r in removed:
            self.cannot_exec.remove(r)
        return policy
    
    def record_power(self, poweron_dg):
        self.total_power.append(poweron_dg)
        
    def power_on(self, file_name, dg_id):
        assert self.archive_server.dgs[dg_id].power_state == "off"
        self.archive_server.dgs[dg_id].power_state = "oning"
        self.archive_server.dgs[dg_id].poweroning_time_stamp = self.time
        with open(f'{file_name}', 'a+') as f:
            f.write(f"At timestamp {self.time:.1f}, issue power-on instruction on AS {self.archive_server.id} DG {dg_id}" + '\n')
    
    def power_off(self, file_name, dg_id):
        assert self.archive_server.dgs[dg_id].power_state == "on"
        self.archive_server.dgs[dg_id].power_state = "offing"
        self.archive_server.dgs[dg_id].poweroffing_time_stamp = self.time
        self.archive_server.dgs[dg_id].on_time.append(self.time - self.archive_server.dgs[dg_id].poweron_time_stamp)
        with open(f'{file_name}', 'a+') as f:
            f.write(f"At timestamp {self.time:.1f}, issue power-off instruction on AS {self.archive_server.id} DG {dg_id}" + '\n')
        self.sche_io.dg_sche_io[dg_id].remove_off_tasks()
        self.archive_server.check_power_state()
        
    def compute_avg_on_time(self):
        avg_on_time = 0
        count = 0
        for j in range(self.hw_config["as_dg_num"]):
            dg = self.archive_server.dgs[j]
            for k in dg.on_time:
                avg_on_time += k
                count += 1
        for dg_id in self.archive_server.poweron_dg:
            avg_on_time += self.time - self.archive_server.dgs[dg_id].poweron_time_stamp
        if count == 0:
            total, num = 0, 0
            for i in range(self.hw_config["as_dg_num"]):
                if self.archive_server.dgs[i].poweron_time_stamp > 0:
                    total += self.time - self.archive_server.dgs[i].poweron_time_stamp
                    num += 1
            if num == 0:
                return 0
            else:
                return total / num
        else:
            return avg_on_time / count
        
    def finish_task(self):
        finish_num = 0
        finish_num += self.sche_io.finish_tasks
        return finish_num
    
    def allocate_task(self, not_archive_t, poweron_dg):
        archive_allocate = []
        if len(poweron_dg) == 0:
            return
        AS_tasks = 0 
        AS_dg_tasks = [0] * len(poweron_dg) 
        for j, dg_id in enumerate(poweron_dg):
            dg = self.archive_server.dgs[dg_id]
            AS_dg_tasks[j] = dg.allocated_tasks
            AS_tasks += self.sche_io.task_num
            
        for t in not_archive_t:
            as_id, dg_id = t.location[0], t.location[1]
            self.sche_io.dg_sche_io[dg_id].set_task(t)
            self.sche_io.task_num += 1
            self.archive_pod.ASs[as_id].dgs[dg_id].allocated_tasks += 1
            if dg_id in poweron_dg:
                index = poweron_dg.index(dg_id)
                AS_dg_tasks[index] += 1
                
        for t in self.archive_tasks:
            as_dg_id = AS_dg_tasks.index(min(AS_dg_tasks))
            dg_id = int(poweron_dg[as_dg_id]) 
            if len(self.sche_io.dg_sche_io[dg_id].pq[4].queue) > MAX_ARCHIVE * 1.5:
                break
            t.allocated = 0
            t.location = [0, dg_id]
            self.sche_io.dg_sche_io[dg_id].set_task(t)
            self.sche_io.task_num += 1
            self.archive_pod.ASs[0].dgs[dg_id].allocated_tasks += 1
            archive_allocate.append(t)
            AS_tasks += 1
            AS_dg_tasks[as_dg_id] += 1
        for t in archive_allocate:
            self.archive_tasks.remove(t)
    
    def update_archive(self, t):
        as_id, dg_id, restore_disk = int(t.location[0]), int(t.location[1]), list(set(t.location[2]) - set(t.parity_location))
        dg = self.archive_server.dgs[dg_id]
        self.archive_pod.write_info.append([self.time, as_id, dg_id, restore_disk, t.parity_location])
        for disk_id in t.location[2]:
            dg.smr_disks[disk_id].replica_queue.append(len(dg.chunk_queue))
        dg.chunk_queue.append([self.time, as_id, dg_id, restore_disk, t.parity_location])
        
    def set_bad_disk(self):
        tasks, remove_record = [], []
        for record in self.bad_records:
            AS_idx, dg_idx, disk_idx, time = record
            if self.time >= time:
                remove_record.append(record)
                dg_chosen = self.archive_pod.ASs[AS_idx].dgs[dg_idx]
                disk_chosen = dg_chosen.smr_disks[disk_idx]
                disk_chosen.flag = 0 # set the disk to bad
                dg_chosen.bad_disks.add(disk_idx)
                dg_chosen.finish_chunk[disk_idx] = 0
                dg_chosen.max_chunk[disk_idx] = len(disk_chosen.replica_queue)
                for j in range(len(disk_chosen.replica_queue)):
                    chunk_id = disk_chosen.replica_queue[j]
                    a_disk, p_disk = dg_chosen.chunk_queue[chunk_id][3], dg_chosen.chunk_queue[chunk_id][4]
                    splice_disk = np.concatenate((a_disk, p_disk)).tolist()
                    for k in list(dg_chosen.bad_disks):
                        if k in splice_disk:
                            splice_disk.remove(k)
                    replica_disks = random.sample(splice_disk, self.hw_config["restore_num"])
                    replica_disks.sort()
                    location = record[:-1]
                    location.append(replica_disks)
                    repair_task = generate_background(self.hw_config, len(self.background_task), self.time, "repair", location, len(dg_chosen.bad_disks))
                    tasks.append(repair_task)
                    self.background_task.append(repair_task)
            else:
                break
        for r in remove_record:
            self.bad_records.remove(r)
        return tasks
    
    def check_scrub(self):
        tasks = []
        for as_id in range(len(self.archive_pod.ASs)):
            for dg_id in range(self.hw_config["as_dg_num"]):
                dg = self.archive_pod.ASs[as_id].dgs[dg_id]
                for chunk_id in range(dg.scrub_id, len(dg.chunk_queue)):
                    last_time = dg.chunk_queue[chunk_id][0]
                    if self.time - last_time > SCRUB_TIME: 
                        a_disk, p_disk = dg.chunk_queue[chunk_id][3], dg.chunk_queue[chunk_id][4]
                        disks_id = np.concatenate((a_disk, p_disk)).tolist()
                        task = generate_background(len(self.background_task), self.hw_config, self.time, "scrub", [as_id, dg_id, chunk_id, disks_id])
                        tasks.append(task)
                        self.background_task.append(task)
                        dg.scrub_id += 1
                    else:
                        break
        return tasks
    
    def get_state(self):
        dg_state = [[] for _ in range(self.hw_config["as_dg_num"])]
        task_size = [[] for _ in range(self.hw_config["as_dg_num"])]
        delay_time = [[] for _ in range(self.hw_config["as_dg_num"])]
        for i in range(self.hw_config["as_dg_num"]):
            dg = self.archive_server.dgs[i]
            dg_sche = self.sche_io.dg_sche_io[i]
            if dg.power_state == "on":
                dg_state[i] = [1, 0, 0, 0, (self.time - dg.poweron_time_stamp) / 1200]
            elif dg.power_state == "oning":
                dg_state[i] = [0, 1, 0, 0, 0]
            elif dg.power_state == "offing":
                dg_state[i] = [0, 0, 1, 0, 0]
            else:
                dg_state[i] = [0, 0, 0, 1, 0]
            task_size[i] = [dg_sche.qt_size[4], dg_sche.qt_size[3], dg_sche.qt_size[2], dg_sche.qt_size[1], dg_sche.qt_size[0], self.sche_io.dg_sche_io[i].qt_size[5]] # arch, resto, scr, repair
            for j in range(len(dg_sche.pq)):
                delay = self.time - dg_sche.pq[j].queue[0].time_stamp if len(dg_sche.pq[j].queue) else 0
                delay_time[i].append(delay)
        dg_state = np.array(dg_state)
        task_size = normalize(task_size)
        delay_time = normalize(delay_time)
        states = np.hstack((dg_state, task_size, delay_time))
        states = states.reshape(1, -1)
        return states
    
    def step(self, actions, index, file_name):
        poweron_dg = list(self.archive_server.poweron_dg)
        poweron_dg.sort()
        withpower_dg = list(self.archive_server.withpower_dg)
        on_with_num = len(poweron_dg) + len(withpower_dg)
        on_dg, off_dg, act_off_dg = [], [], [] 
        policies, on_candidate = [], [] 
        for i, action in enumerate(actions):
            if action < -self.args.act_threshold and [0, i] not in self.cannot_exec and self.archive_server.dgs[i].power_state == "on":
                policies.append(["off", [0, i]])
            elif action > self.args.act_threshold and self.archive_server.dgs[i].power_state == "off":
                on_candidate.append([action, i])
                
        can_on_num = self.hw_config["max_poweron_dg"] - on_with_num
        if can_on_num > 0:
            on_candidate.sort(reverse=True)
            num = 0
            for k in on_candidate:
                policies.append(["on", [0, k[1]]])
                num += 1
                if num == can_on_num:
                    break
                
        for p in policies:
            act, dg_id = p[0], p[1][-1]
            if act == "off":
                if not self.sche_io.dg_sche_io[dg_id].exist_task():
                    off_dg.append(dg_id)
                else:
                    with open(file_name, 'a+') as f:
                        f.write(f"At timestamp {self.time:.1f}, AS {0} DG {dg_id} still has tasks, needs to wait to power-off!")
                        f.write('\n')
                    self.cannot_exec.append([0, dg_id])
                act_off_dg.append(dg_id)
            elif act == "on":
                on_dg.append(dg_id)
                
        done = self.update_state(off_dg, on_dg, index, file_name)
        reward = self.compute_reward(on_with_num, on_dg, act_off_dg)
        next_state = self.get_state()
        return next_state, reward, done
    
    def update_state(self, off_id: list[int], on_id: list[int], index, file_name):
        done = 0
        for dg_id in off_id:
            self.power_off(file_name, dg_id)
        for dg_id in on_id:
            self.power_on(file_name, dg_id)
        self.archive_server.check_power_state()
        # self.record_power(self.archive_server.poweron_dg)
        policy = self.check_clean_up() 
        for p in policy:
            self.power_off(file_name, p[2][0])
            
        for m in range(int(POLICY_INTERVAL / TIMER_INTERVAL)):
            not_archive_t = []
            if len(self.bad_records) and m == 0: 
                new_background_task = self.set_bad_disk()
                not_archive_t.extend(new_background_task)
                
            for i in range(self.task_id, len(self.gen_t)):
                if self.gen_t[i].time_stamp <= self.time or abs(self.gen_t[i].time_stamp - self.time) < 1e-4:
                    self.task_id = i + 1
                    if self.gen_t[i].task_name == "archive":
                        self.archive_tasks.append(self.gen_t[i])
                    else:
                        not_archive_t.append(self.gen_t[i])
                else:
                    break
                
            if self.args.scrub_flag:
                scrub_mod = self.time % SCRUB_INTERVAL
                if scrub_mod < 1e-4 or SCRUB_INTERVAL - scrub_mod < 1e-4:
                    scrub_task = self.check_scrub()
                    not_archive_t.extend(scrub_task)
                    
            if len(not_archive_t) + len(self.archive_tasks) > 0:
                allocate_poweron_dg = copy.deepcopy(list(self.archive_server.poweron_dg))
                for _, dg_id in self.cannot_exec:
                    allocate_poweron_dg.remove(dg_id)
                self.allocate_task(not_archive_t, allocate_poweron_dg)
            
            finished = self.sche_io.update_exec_task(self.cannot_exec)
            for t in finished:
                if t.task_name == "archive":
                    self.update_archive(t)
                    
            self.time += TIMER_INTERVAL
            self.set_time([self.sche_io])
            self.check_all(file_name)
            if self.args.mode == "train":
                if self.time >= self.train_time[index] or abs(self.time - self.train_time[index]) < 1e-4:
                    done = 1
                    break
            elif self.args.mode == "test":
                if self.time >= self.args.max_time or abs(self.time - self.args.max_time) < 1e-4:
                    done = 1
                    break
        return done
        
    def compute_reward(self, on_with_num, on_dg, act_off_dg):
        reward, reward_neg, reward_pos = 0, 0, 0
        re_on_time, re_delay, re_power_dg, re_early_down, re_early_on = 0, 0, 0, 0, 0
        wait_task_flag = True
        re_throughput = self.sche_io.throughput / (RW_SPEED * self.hw_config["smr_num"] * POLICY_INTERVAL / TIMER_INTERVAL)
        re_task = self.sche_io.finishes
        if on_with_num == 0:
            re_power_dg = -2
        elif on_with_num == 1:
            re_power_dg = -1
        total_delay = []
        
        for dg_id in act_off_dg:
            on_time = self.time - POLICY_INTERVAL - self.archive_server.dgs[dg_id].poweron_time_stamp
            if on_time < NEG_ON_TIME:
                re_early_down = max(-10, -400 / (1 + on_time))
        for dg_id in on_dg:
            off_time = self.time - POLICY_INTERVAL - self.archive_server.dgs[dg_id].poweroff_time_stamp
            if off_time < NEG_OFF_TIME:
                re_early_on += max(-8, -400 / (1 + off_time))
                
        for i, dg in enumerate(self.archive_server.dgs):
            if dg.power_state == "on":
                poweron_time = self.time - dg.poweron_time_stamp
                if poweron_time > BASELINE_POWERON_TIME:
                    re_on_t = (poweron_time - BASELINE_POWERON_TIME) / 60
                    re_on_time -= re_on_t
                    
            max_delay = []
            for index, pq_j in enumerate(self.sche_io.dg_sche_io[i].pq):
                q_l = len(pq_j.queue)
                if q_l:
                    wait_task_flag = True
                    delay = (self.time - pq_j.queue[0].time_stamp) / 3600 
                    max_delay.append(delay)
                    total_delay.append([i, index, delay])
            re_delay += -np.sum(max_delay)
        re_delay = re_delay / self.hw_config["as_dg_num"]
        re_task = re_task / 5
        if wait_task_flag and re_throughput == 0: 
            re_throughput = - 0.05
        reward_pos = 0.7 * re_throughput + 0.3 * re_task
        reward_neg = 0.15 * re_on_time + 0.15 * re_delay + 0.1 * re_power_dg + 0.4 * re_early_down + 0.2 * re_early_on
        reward = POS_WEIGHT * reward_pos + NEG_WEIGHT * reward_neg
        self.sche_io.throughput, self.sche_io.finishes = 0, 0
        return reward
    
    def reset(self, flag=True):
        finish_task_num = self.finish_task()
        task_num = self.task_id
        self.metrics.set_task_queue(self.gen_t[:self.task_id+1])
        self.metrics.extend_background(self.background_task)
        self.background_task = []
        with open('./data/bad_record.pkl', 'rb') as f:
            self.bad_records = pickle.load(f)
        if flag:
            results = self.metrics.compute_all(self.time) 
        else:
            results = []
            for t in self.gen_t[:self.task_id+1]:
                t.stripe_num, t.weight = 0, 0
                t.start_timestamp, t.finish_timestamp = None, None
                t.done, t.allocated = False, False
                if t.task_name == "archive":
                    t.location, t.parity_location = None, []
        self.time, self.task_id = 0, 0
        self.archive_server.reset()
        self.archive_pod.reset()
        self.sche_io.reset()
        self.total_power, self.cannot_exec, self.oning_dg, self.archive_tasks = [], [], [], []
        state = self.get_state()
        return state, [results, finish_task_num, task_num]
    