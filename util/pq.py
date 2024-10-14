# -*- encoding: utf-8 -*-
'''
@File : pq.py
@Describe : PRIORITY queue
@Time : 2023/04/16 14:50:46
'''

import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []
        
    def qsize(self):
        return len(self.queue)
    
    def put(self, item):
        heapq.heappush(self.queue, item)

    def get(self):
        return heapq.heappop(self.queue)
    