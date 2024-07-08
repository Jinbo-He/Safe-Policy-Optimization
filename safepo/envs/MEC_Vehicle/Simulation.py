import numpy as np
import random
from collections import deque

class Resource:
    def __init__(self, initial_capacity):
        self.capacity = initial_capacity

    def add_resource(self, amount):
        self.capacity += amount

    def remove_resource(self, amount):
        self.capacity -= amount
        return True

    def update_resource(self, new_capacity):
        self.capacity = new_capacity

    def get_resource(self):
        return self.capacity


class Task:
    def __init__(self, from_id, required_resource, size, delay, start_steps=0):
        self.from_id = from_id
        self.required_resource = required_resource  # task is a dictionary with 'id' and 'resource_needed'
        self.delay = delay  # delay is the time the task has been waiting in the queue
        self.size = size
        self.start_steps = 0
        self.proc_steps = 0
        self.alloc_res = 0
        self.trans_time = 0
        self.trans_steps = 0
        self.reward = 0
        self.to_id = None
        self.min_res_alloc = required_resource / delay
    
    @property
    def info(self):
        return [self.required_resource, self.size, self.delay]
        
        
class MEC:
    def __init__(self, mec_id, nei_size, grid_size, initial_resource_capacity):
        self._self_trans_tasks = []
        self.mec_id = mec_id
        self.resource = Resource(initial_resource_capacity)
        self.neighbor_size = nei_size
        self.grid_size = grid_size
        self.neighbors = []  # list of neighboring MECs
        self.neighbors_ids = self.__calculate_neighbors(mec_id)
        self.current_vehicles = None
        self.on_processing_tasks = []
        self.res_alloc = None
        self._trans_tasks = []
        self.current_tasks = []
    
    @property
    def res(self):
        return self.resource.get_resource()
    
    @property
    def tasks_info(self) -> list:
        return [task.info for task in list(self.current_tasks)]
    
    @property
    def trans_tasks(self):
        return self._trans_tasks
    
    @property
    def self_trans_tasks(self):
        return self._self_trans_tasks

    def receive_task(self, task):
        if len(self.on_processing_tasks) < 20:
            if self.res > task.alloc_res:
                self.on_processing_tasks.append(task)
                self.resource.remove_resource(task.alloc_res)
                return True
        return False
    
    def append_task(self, task):
        self.current_tasks.append(task)
    
    def reset_task(self):
        self.current_tasks.clear()
        
    def __calculate_neighbors(self, mec_id) -> list:
        neighbors = []
        grid_size = self.grid_size
        row = mec_id // grid_size
        col = mec_id % grid_size
        
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < grid_size and 0 <= j < grid_size:
                    neighbors.append(i * grid_size + j)
                else:
                    neighbors.append(-1)  # -1 represents no neighbor (out of bounds)
        return neighbors
    
    def __divide_resources(self):
        total_resource_needed = sum(task.required_resource for task in self.on_processing_tasks)
        divided_resources = [self.res_alloc * (task.required_resource / total_resource_needed) for task in
                             self.on_processing_tasks]
        return divided_resources
    
    def receive_trans_tasks(self, timestep):
        for task in self.trans_tasks:
            if task.stat_steps + task.proc_steps == timestep:
                flag = self.receive_task(task)
                
    def finish_tasks(self, timestep):
        finished_tasks = [task for task in self.on_processing_tasks if
                          task.start_steps + task.proc_steps + task.trans_steps == timestep]
        for task in finished_tasks:
            self.resource.add_resource(task.alloc_res)
            self.on_processing_tasks.remove(task)
        
    def reshape_task(self):
        tmp = []
        
if __name__ == '__main__':
    MECs = [MEC(i, 3, 3, 100) for i in range(9)]
    print(8)
