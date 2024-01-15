# -*- encoding: utf-8 -*-
'''
@File : archive_server.py
@Describe : archive server class
@Time : 2022/07/19 15:22:25
'''

from structure.config import HDD_SIZE
from structure.disk import Disk, DiskGroup


class Archive_Server:
    '''
        id: server id
        hdd: All hdds in an archive server
        dgs: The diskgroups an AS manage, set 9 in default
        total_size: the storage space, except hdds
        cur_size: current size of the archive server
        time: current time
        poweron_dg: current power-on disk group
    '''
    def __init__(self, id: int, dgs: list['DiskGroup'], hw_config):
        self.id = id
        self.dgs = dgs
        self.hw_config = hw_config
        self.hdd = [Disk(id=i, total_size=HDD_SIZE, cur_size=0, flag=1) for i in range(hw_config["hdd_num"])]
        self.total_size = self.dgs[0].total_size * len(self.dgs)
        self.cur_size = 0
        self.total_chunk_num = self.total_size / hw_config["chunk_size"]
        self.poweron_dg = set()
        self.withpower_dg = set()
        
    def reset(self):
        for dg in self.dgs:
            self.cur_size += dg.reset()
        self.poweron_dg, self.withpower_dg = set(), set()
        
    def get_cur_size(self):
        cur_size = 0
        for dg in self.dgs:
            cur_size += dg.get_cur_size()
        self.cur_size = cur_size
        return cur_size
    
    def check_power_state(self):
        self.poweron_dg = set()
        self.withpower_dg = set()
        for dg in self.dgs:
            if dg.power_state == "on":
                self.poweron_dg.add(dg.id)
            if dg.power_state == "oning" or dg.power_state == "offing":
                self.withpower_dg.add(dg.id)
        assert len(self.poweron_dg) + len(self.withpower_dg) <= self.hw_config["max_poweron_dg"]
        