#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File      : metircs.py
@Describe  : Compute evaluation metrics
@Time      : 2022/08/01 14:47:37
'''


class Metrics:
    def __init__(self):
        self.task_queue = []
        self.latency = {}
        self.completion_time = {}
        self.delay_time = {}
        self.service_time = {}

    def set_task_queue(self, task_queue):
        self.task_queue = task_queue

    def extend_background(self, background_task):
        self.task_queue.extend(background_task)

    def compute_all(self, time):
        throughput = self.compute_all_metrics(time)
        avg_latency = self.avg_latency()
        avg_complete_time, avg_delay_time, avg_service_time = self.avg_metrics()
        return [avg_latency, avg_complete_time, avg_delay_time, avg_service_time, throughput]
    
    def compute_all_metrics(self, time):
        workload = 0
        archive_workload = 0
        restore_workload = 0
        scrub_workload = 0
        repair_workload = 0
        for t in self.task_queue:
            workload += t.task_size if t.done else t.stripe_num
            if t.task_name == "archive":
                archive_workload += t.stripe_num
            elif t.task_name == "restore":
                restore_workload += t.stripe_num
            elif t.task_name == "scrub":
                scrub_workload += t.stripe_num
            else:
                repair_workload += t.stripe_num
            if t.done:
                self.latency[t.task_id] = t.start_timestamp - t.time_stamp 
                self.completion_time[t.task_id] = t.finish_timestamp - t.time_stamp 
                self.delay_time[t.task_id] = t.start_timestamp - t.time_stamp 
                self.service_time[t.task_id] = t.finish_timestamp - t.start_timestamp 
            else:
                self.latency[t.task_id] = time - t.time_stamp
            # reset task
            t.stripe_num, t.weight = 0, 0
            t.start_timestamp, t.finish_timestamp = None, None
            t.done, t.allocated = False, False
            if t.task_name == "archive":
                t.location, t.parity_location = None, []

        throughput = workload / (time - 60) / 128
        archive_throughput = archive_workload / (time - 60) / 128
        restore_throughput = restore_workload / (time - 60) / 128
        scrub_throughput = scrub_workload / (time - 60) / 128
        repair_throughput = repair_workload / (time - 60) / 128
        return [throughput, archive_throughput, restore_throughput, scrub_throughput, repair_throughput]
    
    def compute_tail_latency(self):
        latencys = list(self.latency.values())
        latencys.sort()
        tail_99_latency = latencys[int(len(latencys) * 0.99)]
        tail_latency = latencys[-1]
        return tail_99_latency, tail_latency

    def avg_latency(self):
        total = 0
        count = 0
        for k in self.latency.keys():
            total += self.latency[k]
            count += 1
        if count == 0:
            return 0
        return total / count


    def avg_metrics(self):
        com_total, delay_total, service_total = 0, 0, 0
        count = 0
        for k in self.completion_time.keys():
            com_total += self.completion_time[k]
            delay_total += self.delay_time[k]
            service_total += self.service_time[k]
            count += 1
        if count == 0:
            return 0, 0, 0
        else:
            return com_total / count, delay_total / count, service_total / count

    # def record_all_tasks(self):
    #     task_dg = [[0 for _ in range(AS_DG_NUM)] for _ in range(AS_NUM)]
    #     for t in self.task_queue:
    #         if not t.location:
    #             continue
    #         as_id, dg_id = t.location[0], t.location[1]
    #         task_dg[as_id][dg_id] += 1
    #     return task_dg
        

# save metrics result
class Result:
    def __init__(self, results, avg_on_time, urgent, medium, low):
        self.latency = results[0]
        self.completion_time = results[1]
        self.delay_time = results[2]
        self.service_time = results[3]
        self.throughput = results[-1][0]
        self.archive_throughput = results[-1][1]
        self.restore_throughput = results[-1][2]
        self.scrub_throughput = results[-1][3]
        self.repair_throughput = results[-1][4]
        self.avg_poweron_time = avg_on_time
        self.urgent = urgent
        self.medium = medium
        self.low = low
        