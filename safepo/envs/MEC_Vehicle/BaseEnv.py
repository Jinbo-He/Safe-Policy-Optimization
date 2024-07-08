import random
import time
from copy import deepcopy
from typing import Tuple
import math
import numpy as np
import numpy.random
# import traci
import math
import yaml
from gym.spaces import Box, Discrete

from harl.envs.MEC_Vehicle.Simulation import MEC, Task
from collections import deque
import libsumo as traci


def calculate_distance(x1, y1, x2, y2):
	return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_closest_center(x, y, centers):
	return min(centers, key=lambda center: calculate_distance(x, y, centers[center][0], centers[center][1]))


def allocate_tasks(allocation_list, tasks):
	grid_size = len(allocation_list)
	node_task_list = {i: [] for i in range(grid_size)}
	
	# 计算每个节点应分配的任务数量，并向下取整
	total_tasks = len(tasks)
	node_task_count = {i: int(round(total_tasks * allocation_list[i][0])) for i in range(grid_size)}
	
	# 计算已分配任务总数
	assigned_tasks = sum(node_task_count.values())
	
	# 如果分配的任务数少于总任务数，将剩余任务按原比例顺序分配
	remaining_tasks = total_tasks - assigned_tasks
	if remaining_tasks > 0:
		sorted_indices = sorted(range(grid_size), key=lambda x: -allocation_list[x][0])  # 按比例从大到小排序
		for i in sorted_indices:
			if remaining_tasks > 0:
				node_task_count[i] += 1
				remaining_tasks -= 1
			else:
				break
	
	# 按节点分配任务
	task_index = 0
	for node in range(grid_size):
		for _ in range(node_task_count[node]):
			if task_index < total_tasks:
				node_task_list[node].append(tasks[task_index])
				task_index += 1
	
	# 对每个节点的任务按资源需求分配比例
	for node in range(grid_size):
		tasks_in_node = node_task_list[node]
		if tasks_in_node:
			total_require_resource = sum(task.min_res_alloc for task in tasks_in_node)
			if total_require_resource > 0:  # 防止除以零
				for task in tasks_in_node:
					task.alloc_res = (task.min_res_alloc / total_require_resource) * allocation_list[node][1]
	
	# 构建结果列表
	result = [node_task_list[i] for i in range(grid_size)]
	
	return result


class Environment:
	def __init__(self, args):
		# Extract the values
		self.suc_count = 0
		self.episode = 500
		self.grid_size = args['environment_config']['grid_size']
		self.neighborhood_size = args['environment_config']['neighborhood_size']
		self.resource_capacity = args['environment_config']['resource_capacity']
		self.task_res_info = args['environment_config']['task_res_info']
		self.task_size_info = args['environment_config']['task_size_info']
		self.task_delay_info = args['environment_config']['task_delay_info']
		self.transmission = args['environment_config']['transmission']
		self.n_agents = self.grid_size ** 2
		self.mecs = None
		self.pre_state = None
		self.state = None
		self.pre_obs = None
		self.obs = None
		self.centers = {f"agent_{i * 5 + j}": (75 + i * 150, 75 + j * 150) for i in range(5) for j in range(5)}
		self.n_agents = self.grid_size ** 2
		self.pre_vehicle = None
		self.num_vehicle = None
		self.center_vehicle_count = None
		self.timestep = 0
		
		self.obs_space = self.grid_size ** 2 * 2 + 10 * 3
		self.observation_space = [np.zeros(shape=self.obs_space)] * self.n_agents
		# self.action_space = [MultiDiscrete([self.depth * self.depth, self.percentage])] * self.n_agents
		type_ = args["environment_config"]['type']
		if type_ == 'hybrid':
			self.action_space = [(Discrete(self.grid_size * self.grid_size), Box(0, 1, shape=(3,)))] * self.n_agents
		else:
			self.action_space = [Box(0, 1, shape=(9 * 2,))] * self.n_agents
		self.share_observation_space = [np.zeros(shape=self.obs_space * self.n_agents)]
		self.avg_success = deque()
		self.task_buffer = {}
	
	def reset(self):
		try:
			traci.close()
		except Exception as e:
			pass
		self.timestep = 0
		sumo_binary = "/home/jinbo/Downloads/sumo-1.20.0"
		config_file = "/home/jinbo/Repos/harl_git/harl/envs/MEC_Vehicle/simulation/grid.sumocfg"
		traci.start([sumo_binary, "-c", config_file, '--seed', str(numpy.random.randint(0, 10000)), '--no-step-log', ])
		# traci.simulationStep()
		# print(traci.simulation.getDeltaT())
		# print(traci.vehicle.getIDList())
		for i in range(100):
			traci.simulationStep()  # 执行1000步模拟=100s
		# print(f"pre sim finished at {traci.simulation.getTime()}")
		for i in range(self.n_agents):
			self.task_buffer[f'agent_{i}'] = {}
		self.state = self.init_mecs()
		self.pre_obs, self.obs = self.get_obs()
		self.state = self.get_input_state()
		return self.obs, self.state, None
	
	def act(self, actions):
		for agent_id in range(self.n_agents):
			action = actions[agent_id]
			target_id = int(action[0])
			target_ratio = action[1]
			res_self = action[2]
			res_target = action[3]
			
			mec = self.mecs[agent_id]
			task_info = mec.current_tasks
			# lenns += len(task_info)
			# print(f'task num {len(task_info)}')
			allocate_list = [(0, 0)] * self.grid_size ** 2
			if target_id == agent_id:
				ratio = res_target + res_self
				ratio = ratio if ratio <= 1 else 1
				allocate_list[agent_id] = [1, ratio]
			else:
				allocate_list[target_id] = [target_ratio, res_target]
				allocate_list[agent_id] = [1 - target_ratio, res_self]
			
			final_decision = allocate_tasks(allocate_list, task_info)
			for decision_id, decision in enumerate(final_decision):
				if not decision == []:
					distance = self.obs[agent_id][decision_id * 2 + 1]
					trans_time = [distance * task.size / self.transmission for task in decision]
					trans_steps = [math.floor(trans/0.1) for trans in trans_time]
					append_list = []
					for task, trans, step in zip(decision, trans_time, trans_steps):
						task.to_id = decision_id
						task.trans_time = trans
						task.trans_steps = step
						task.start_steps = self.timestep
						task.alloc_res = task.alloc_res * self.mecs[agent_id].res
						append_list.append(task)
					self.mecs[decision_id].trans_tasks.extend(append_list)
					
	def proc(self, rewards):
		for index in range(self.n_agents):
			mec = self.mecs[index]
			# finish tasks in MECs
			# receiving trans tasks
			arrived_task = [task for task in mec.trans_tasks if task.start_steps + task.trans_steps == self.timestep]
			for task in arrived_task:
				if task.min_res_alloc > task.alloc_res:
					pass
				# 	rewards[task.from_id] -= 1
				# 	task.reward = 1e-5
				# elif task.required_resource / task.alloc_res < 0.05:
				# 	# rewards[task.from_id] -= 1
				# 	task.reward = 1e-5
				else:
					delay = task.trans_time + task.required_resource / task.alloc_res
					if delay > task.delay:
						task.reward = 1e-5
					else:
						flag = mec.receive_task(task)
						proc_time = task.required_resource / task.alloc_res
						proc_steps = math.floor(proc_time/0.1)
						if flag:
							task.proc_steps = proc_steps
							task.reward = 1 / (1.5 * task.trans_time + proc_time)
							rewards[task.from_id] += task.reward
							self.suc_count += 1
						# print(f'task {task.from_id} rew{rewards[task.from_id]}')
						else:
							# rewards[task.from_id] -= 1
							task.reward = 1e-5
							# count += 1
				task_timestep = self.task_buffer[f'agent_{index}'][f'step_{task.start_steps}']
				if task.to_id != task.from_id:
					filter_list = [task for task in task_timestep if task.to_id != task.from_id]
				else:
					filter_list = [task for task in task_timestep if task.to_id == task.from_id]
				rew = 0
				for task_ in filter_list:
					if task_.reward != 0:
						rew += task_.reward/len(filter_list)
					else:
						rew = 0
						break
				rewards[task.from_id] += rew
			# print(f'rest {index} :{self.mecs[index].res}')
			mec.finish_tasks(self.timestep)
			# print(f'rest {index} :{self.mecs[index].res}')
		return rewards
	
	def step(self, actions):
		# 执行一个模拟步骤
		# Reset vehicle count for each center
		# print(traci.simulation.getTime())
		rewards = np.zeros(shape=(self.n_agents, 1))
		lenns = 0
		self.act(actions)
		# print(f'task_num {task_count}')
		# 做完决策之后处理后续
		rewards = self.proc(rewards)
		# update obs state... etc
		# print the rest res
		# for i in range(self.n_agents):
		# 	print(f'rest {i} :{self.mecs[i].res}')
		rew = rewards.sum(keepdims=True)/self.n_agents
		if self.timestep <= self.episode - 3:
			self.avg_success.append([self.suc_count, self.num_vehicle])
		else:
			self.avg_success.append([self.suc_count, 0])
		# print(suc_count/self.num_vehicle)
		traci.simulationStep()
		self.timestep += 1
		# self.pre_vehicle = self.center_vehicle_count
		self.center_vehicle_count = self.get_num_vehicle()
		self.generate_tasks()
		self.pre_obs, self.obs = self.get_obs()
		self.state = self.get_input_state()
		info = [{}]
		self.suc_count = 0
		if self.timestep == self.episode:
			dones = [True] * self.n_agents
			self.timestep = 0
			# print(f' one episode finished at {traci.simulation.getTime()}')
			traci.close()
			count_list = list(self.avg_success)
			avg_count, avg_task_count = sum(sublist[0] for sublist in count_list), sum(
				sublist[1] for sublist in count_list)
			# print(f'avg success count: {avg_count}')
			# print(f'avg task count: {avg_task_count}')
			print(f'avg success rate: {avg_count / avg_task_count}')
			info = [{'avg_success_rate': avg_count / avg_task_count}]
			self.avg_success.clear()
		else:
			dones = [False] * self.n_agents
		return self.obs, self.state, rew, dones, info, None
	
	# generate new tasks
	
	def init_mecs(self) -> list[MEC]:
		mecs = [MEC(i * self.grid_size + j, self.neighborhood_size, self.grid_size,
		            np.random.randint(self.resource_capacity[0], self.resource_capacity[1]))
		        for j in range(self.grid_size) for i in range(self.grid_size)]
		self.mecs = mecs
		self.center_vehicle_count = self.get_num_vehicle()
		self.generate_tasks()
		return mecs
	
	def generate_tasks(self):
		for agent, item in self.center_vehicle_count.items():
			agent_id = int(agent.split('_')[1])
			mec = self.mecs[agent_id]
			mec.reset_task()
			# make sure at least 1 task
			for i in range(item) if item != 0 else range(1):
				if i == 10:
					break
				res = np.random.randint(self.task_res_info[0], self.task_res_info[1])
				size = np.random.randint(self.task_size_info[0], self.task_size_info[1])
				delay = np.random.uniform(self.task_delay_info[0], self.task_delay_info[1])
				task = Task(agent_id, res, size, delay, self.timestep)
				mec.append_task(task)
				self.task_buffer[f'agent_{agent_id}'].setdefault(f'step_{self.timestep}', [])
				self.task_buffer[f'agent_{agent_id}'][f'step_{self.timestep}'].append(task)
				# 这里必须赋值task，而不是直接赋值mec.current_tasks，否则会出现引用问题
			
	def get_num_vehicle(self) -> dict:
		center_vehicle_count = {center: 0 for center in self.centers}
		# 遍历每辆车，更新车辆计数
		for vehicle_id in traci.vehicle.getIDList():
			x, y = traci.vehicle.getPosition(vehicle_id)
			closest_center_id = get_closest_center(x, y, self.centers)
			center_vehicle_count[closest_center_id] += 1
		self.num_vehicle = sum(center_vehicle_count.values())
		return center_vehicle_count
	
	def get_obs(self) -> Tuple[list, list]:
		# pre_obs = deepcopy(self.obs)
		obs = []
		for i in range(self.n_agents):
			mec = self.mecs[i]
			tmp = []
			for neighbors_id in range(self.n_agents):
				tmp.append(self.mecs[neighbors_id].res)
				# 模拟随机路由
				x, y = divmod(i, self.grid_size)
				nei_x, nei_y = divmod(neighbors_id, self.grid_size)
				distance = abs(x - nei_x) + abs(y - nei_y)
				tmp.append(distance)
			for info in mec.tasks_info:
				tmp.extend(info)
			while len(tmp) < 2 * self.grid_size ** 2 + 10 * 3:
				tmp.extend([0])
			obs.append(tmp)
		return [], obs
	
	def get_input_state(self):
		return [np.array(self.obs).reshape(-1)]
	
	def seed(self, _seed):
		np.random.seed(_seed)
	
	def close(self):
		traci.close()


if __name__ == '__main__':
	traci.start(
		["sumo", "-c", "/home/jinbo/Repos/harl_git/harl/envs/MEC_Vehicle/simulation/grid.sumocfg", '--no-step-log'])
	while traci.simulation.getMinExpectedNumber() > 0:
		traci.simulationStep()
	traci.close()
