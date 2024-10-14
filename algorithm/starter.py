# -*- encoding: utf-8 -*-
'''
@File : starter.py
@Describe : starter for RL training
'''

from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from algorithm.algo_config import INSTANCES, BUFFER_INTERVAL
from algorithm.PPO import PPOAgent
from structure.config import POLICY_INTERVAL, RANDOM_SEED
from structure.timer import Timer
from structure.archive_pod import Archive_Pod
from structure.scheduler import Sche_IO
from util.metrics import Metrics
from util.utils import set_seed, print_begin, print_eval, print_test, print_reset, test_seed, remove_test_seed

# from [0,1] to [-max_action, max_action]
def Action_adapter(a, max_action):
    return 2 * (a - 0.5) * max_action

class Starter:
    def __init__(self, args, timer: Timer, archive_pod: Archive_Pod, agent: PPOAgent, sche_io: Sche_IO, metrics: Metrics, folder_name, interval=POLICY_INTERVAL):
        self.args = args
        self.timer = timer
        self.archive_pod = archive_pod
        self.agent = agent
        self.sche_io = sche_io
        self.metrics = metrics
        self.interval = interval
        self.folder_name = folder_name
        self.log_folder = f"./print_log/{folder_name}"
        self.checkpoint_folder = f"./checkpoint/{folder_name}"
        self.train_name = self.log_folder + "/train.txt"
        self.eval_name = self.log_folder + "/eval.txt"
        self.reward_name = self.log_folder + "/reward.txt"
        self.checkpoint_folder = self.checkpoint_folder
        
    def evaluate_model(self, index, state, evaluate_num, name="eval"):
        if name == "eval":
            write_name = self.eval_name
        elif name == "test":
            write_name = self.log_folder + "/test.txt"
        else:
            raise ValueError("name must be eval or test")
        test_seed(random_seed=RANDOM_SEED)
        done = False
        evaluate_reward = 0
        while not done:
            act = self.agent.policy.evaluate(state)
            action = Action_adapter(act, 1)
            state, r, done = self.timer.step(action, index, write_name)
            evaluate_reward += r
        run_time = self.timer.time
        self.archive_pod.reset()
        state, [results, finish_task_num, task_num] = self.timer.reset()
        if name == "eval":
            print_eval(write_name, results, finish_task_num, task_num, evaluate_num, run_time, evaluate_reward)
        elif name == "test":
            latencies = self.timer.metrics.compute_tail_latency()
            print_test(write_name, results, finish_task_num, task_num, evaluate_num, run_time, evaluate_reward, latencies)
        remove_test_seed()
        return evaluate_reward
    
    def begin(self, file_name):
        self.timer.time = 60.0
        self.timer.set_time([self.timer.sche_io])
        max_poweron_num = self.timer.hw_config["max_poweron_dg"]
        for i in range(max_poweron_num):
            dg = self.timer.archive_server.dgs[i]
            dg.poweron_time_stamp = 60.0
            dg.power_state = "on"
        self.timer.check_all(file_name)
        set_seed()
        print_begin(file_name, max_poweron_num)
        
        
    def run(self):
        state, _ = self.timer.reset()
        # self.begin(self.train_name)
        summary_writer = SummaryWriter(log_dir=f"train/{self.folder_name}", comment="train from scratch")
        print_running_reward, max_r = 0, 0
        episode_num, episode_reward, index, evaluate_num = 1, 0, 0, 0
        evaluate_rewards, episode_rewards = [], []  # episode_rewards saves each episode reward of the same time length 
        rewards = [] 
        train_episode = [0 for _ in range(len(self.timer.train_time))] 
        time_step = 0
        print_freq = max(5, int(5 * self.args.update_timestep / ((self.timer.train_time[index] - 60))))  # print avg reward in the episode
        update_buffer_step = 0  # to update episode_buffer
        traj_state, traj_action, traj_reward = [], [], []
        
        ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------")
        print("max training timesteps: ", INSTANCES)
        print("model saving frequency: " + str(self.args.save_timestep) + " timesteps")
        start_time = datetime.now().replace(microsecond=0)
        # reward_scaling = RewardScaling(shape=1, gamma=GAMMA)
        # reward_scaling.reset()
        
        for i in range(INSTANCES):
            if i > 0 and i % self.args.save_timestep == 0:
                print("Save param at time step : ", time_step)
                self.agent.save(f"{self.checkpoint_folder}/{time_step}")
                
            if index >= len(self.timer.train_time):
                break

            act, logprob_a = self.agent.policy.select_action(state)
            action = Action_adapter(act, 1)
            next_state, reward, done = self.timer.step(action, index, self.train_name)

            # save trajectory for updating episode buffer
            traj_state.append(state.squeeze().tolist())  # list, not tensor   
            traj_action.append(action)
            traj_reward.append(reward)
            update_buffer_step += 1
            if update_buffer_step > 0 and update_buffer_step % BUFFER_INTERVAL == 0:
                self.agent.update_episode_buffer(traj_state, traj_action, traj_reward)
                traj_state, traj_action, traj_reward = [], [], []
            
            self.agent.put_data((state, act, reward, next_state, logprob_a, done)) 
            state = next_state
            time_step += 1
            episode_reward += reward
            
            if time_step % self.args.update_timestep == 0:
                self.agent.update(summary_writer)
                set_seed()
                
            if done:
                rewards.append(episode_reward)
                train_episode[index] += 1
                print_running_reward += episode_reward
                episode_rewards.append(episode_reward)
                
                # printing average reward
                if episode_num % print_freq == 0:
                    print_avg_reward = print_running_reward / print_freq
                    print_avg_reward = round(print_avg_reward, 2)
                    with open(self.reward_name, 'a+') as f:
                        f.write(f"Episode : {episode_num} Length : {self.timer.train_time[index]} Timestep : {time_step} Average Reward :{print_avg_reward}" + '\n')
                    print_running_reward = 0
                    
                state, [results, finish_task_num, task_num] = self.timer.reset()
                print_reset(self.train_name, results, finish_task_num, task_num, episode_num, self.timer.train_time[index], episode_reward)
                summary_writer.add_scalar("Episode reward", episode_reward, episode_num)
                summary_writer.add_scalar("Total throughput", results[-1][0], episode_num)
                
                # Evaluate the policy
                if episode_num % self.args.eval_episode == 0:
                    # self.begin(self.eval_name)
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_model(index, state, evaluate_num)
                    if evaluate_reward >= max_r:
                        self.agent.save(f"{self.checkpoint_folder}/best")
                        print("Save best param at time step: ", time_step)
                    max_r = max(max_r, evaluate_reward)
                    summary_writer.add_scalar('Eval_rewards', evaluate_reward, global_step=time_step)
                    print("evaluate_num:{} \t reward:{:.2f} \t".format(evaluate_num, evaluate_reward))
                    evaluate_rewards.append(evaluate_reward)

                # increase running ts
                if len(episode_rewards) >= 50 and abs(episode_reward - np.mean(episode_rewards[-10:])) < 2:
                    index += 1
                    episode_rewards = []
                    
                episode_reward = 0
                episode_num += 1
                # self.begin(self.train_name)
                
        summary_writer.close()
        print("=======================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time : ", end_time - start_time)
        print("=======================================================================================")
            
    def test(self):
        index = 0
        state, _ = self.timer.reset()
        model_name = './checkpoint/' + f'{self.args.max_time}_{self.args.workload_rate}_{self.args.archive_rate}/' + f'{self.args.model_name}'
        self.agent.load(model_name)
        self.begin(self.log_folder+"/test.txt")
        evaluate_reward = self.evaluate_model(index, state, evaluate_num=1, name="test")
        print("------------------------------------------------")
        print(f"Time: {self.args.max_time}, workload rate: {self.args.workload_rate}r/s, reward : {evaluate_reward:.2f}")
        print(f"More details in {self.log_folder}/test.txt")
        