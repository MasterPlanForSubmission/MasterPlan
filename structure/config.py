# -*- encoding: utf-8 -*-
'''
@File : config.py
@Describe : basic config settings
@Time : 2022/07/19 14:11:35
'''

###### simulator parameters ######
PLOT_ALL = False 
TIMER_INTERVAL = 0.1 # interval time of the timer
RANDOM_SEED = 42
LATENCY = 4.16 * 1e-3 
POLICY_INTERVAL = 5 
DISK_LIST_NUM = 10000 # generate possible archive disk list record

###### archive pod parameters ######
ZONE_SIZE = 256 # a zone is 256MB
SMR_SIZE = 14 * (1024**2) # size of a smr disk is 14TB, calculate in MB
HDD_SIZE = 12 * (1024**2) # size of a hdd disk is 12TB, calculate in MB
POWER_STATE = ["off", "offing", "oning", "on"]
TIME_PREVENT_OFF = 5 * 60 
TIME_PREVENT_ON = 60 * 60 
ONING_TIME = 60 
OFFING_TIME = 30 

###### system parameters #######
BASELINE_POWERON_TIME = 20 * 60 + 30
RW_SPEED = 50 * TIMER_INTERVAL # read or write speed of a singledisk
REPAIR_TIME = 3 * 24 * 3600 # recommended repair time is 2-5 days, set 3 days in default
SCRUB_INTERVAL = 30 * 60 
SCRUB_TIME = 24 * 3600 
PRIORITY = [1*60*60, 4*60*60, 12*60*60] # Urgent, Medium, Low
PRIORITY_NAMES = {3600: "urgent", 4*60*60: "medium", 12*60*60: "low"}
COMMON_POSS = [0.1, 0.4, 0.5]
URGENT_POSS = [0.8, 0.1, 0.1]
    
###### task parameters ######
MAX_ARCHIVE = 6 
MAX_RESTORE = 20 
MAX_REPAIR = 3
MAX_SCRUB = 3 
REPAIR_WEIGHT = {REPAIR_TIME: 4, 2 * REPAIR_TIME / 3: 8, REPAIR_TIME / 3: 12}
CLASS_IDX = {"scrub": 0, "restore_low": 1, "restore_medium": 2, "restore_urgent": 3, "archive": 4, "repair": 5}
LIST_PRIORITY = [12*3600, 12*3600, 4*3600, 3600, 4*3600, 4*3600]
LIST_WEIGHT = [1, 1, 4, 12, 4, 4]
PQ_NUM = len(LIST_WEIGHT) 
