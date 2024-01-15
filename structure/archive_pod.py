# -*- encoding: utf-8 -*-
'''
@File : archive_pod.py
@Describe : An archive pod of all disks
@Time : 2023/02/06 21:54:36
'''
from structure.archive_server import Archive_Server

class Archive_Pod:
    def __init__(self, as_list: list[Archive_Server]):
        self.ASs = as_list
        self.space_size = self.get_space_size()
        self.write_info = []
        
    def reset(self):
        self.space_size = self.get_space_size()
        self.write_info = []
        
    # record chunk list
    def record_write(self):
        for as_id in range(len(self.ASs)):
            for dg_id in range(len(self.ASs[0].dgs)):
                self.write_info.extend(list(self.ASs[as_id].dgs[dg_id].chunk_queue))
                
    def get_space_size(self):
        space_size = []
        for as_i in self.ASs:
            space_size.append(as_i.get_cur_size())
        self.space_size = space_size
        return space_size
    