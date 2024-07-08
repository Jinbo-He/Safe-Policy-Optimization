import yaml

import BaseEnv

def read_config(filepath):
	file = open(filepath)
	config = yaml.safe_load(file)
	return config

def rule():
	pass

def run():
	args = read_config('config.yaml')
	env = BaseEnv.Environment(args)
	env.reset()
	fixed_action = []
	for i in range(25):
		fixed_action.append([i, 0, 1, 0])
	for i in range(500):
		if i == 499:
			obs, state, reward, done, info, _ = env.step(fixed_action)
		else:
			obs, state, reward, done, info, _ = env.step(fixed_action)
		print(reward)
if __name__ == '__main__':
	for i in range(3):
		run()