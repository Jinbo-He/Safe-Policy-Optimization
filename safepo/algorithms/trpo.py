# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import argparse
import random
import time
import numpy as np
import safety_gymnasium
import torch
import torch.optim
import torch.nn as nn

from collections import deque
from distutils.util import strtobool
from typing import Callable
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeNormalizeObservation, SafeUnsqueeze, SafeRescaleAction
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Distribution
from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.model import ActorVCritic
from safepo.common.logger import EpochLogger


def parse_args():
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--device", type=str, default="cpu",
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--torch-threads", type=int, default=1,
        help="number of threads for torch")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--total-steps", type=int, default=1024000,
        help="total timesteps of the experiments")
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0",
        help="the id of the environment")
    # general algorithm parameters
    parser.add_argument("--steps_per_epoch", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--update-iters", type=int, default=40,
        help="the max iteration to update the policy")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--target-kl", type=float, default=0.02,
        help="the target KL divergence threshold")
    parser.add_argument("--max-grad-norm", type=float, default=40.0,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--critic-norm-coef", type=float, default=0.001,
        help="the critic norm coefficient")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--lam", type=float, default=0.95,
        help="the lambda for the reward general advantage estimation")
    parser.add_argument("--lam-c", type=float, default=0.95,
        help="the lambda for the cost general advantage estimation")
    parser.add_argument("--standardized_adv_r", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles reward advantages standardization")
    parser.add_argument("--standardized_adv_c", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles cost advantages standardization")
    parser.add_argument("--actor_lr", type=float, default=3e-4,
        help="the learning rate of the actor network")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
        help="the learning rate of the critic network")
    parser.add_argument("--linear-lr-decay", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles learning rate annealing for policy and value networks")
    # logger parameters
    parser.add_argument("--log-dir", type=str, default="../runs",
        help="directory to save agent logs (default: ../runs)")
    parser.add_argument("--use-tensorboard", type=lambda x: bool(strtobool(x)), default=False,
        help="toggles tensorboard logging")
    # algorithm specific parameters
    parser.add_argument("--fvp-sample-freq", type=int, default=1,
        help="the sub-sampling rate of the observation")
    parser.add_argument("--cg-damping", type=float, default=0.1,
        help="the damping value for conjugate gradient")
    parser.add_argument("--cg-iters", type=int, default=15,
        help="the number of conjugate gradient iterations")
    args = parser.parse_args()
    return args
    
def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, 'No gradients were found in model parameters.'
    return torch.cat(flat_params)

def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x

def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'

def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, 'No gradients were found in model parameters.'
    return torch.cat(grads)

def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    ) -> torch.Tensor:
    policy.actor.zero_grad()
    q_dist = policy.actor(fvp_obs)
    with torch.no_grad():
        p_dist = policy.actor(fvp_obs)
    kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

    grads = torch.autograd.grad(kl,tuple(policy.actor.parameters()),create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_p = (flat_grad_kl * params).sum()
    grads = torch.autograd.grad(
        kl_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    logger.store(
        **{
            'Train/KL': kl.item(),
        },
    )
    return flat_grad_grad_kl + params * args.cg_damping

def search_step_size(
    policy: ActorVCritic,
    step_direction: torch.Tensor,
    grads: torch.Tensor,
    p_dist: Distribution,
    obs: torch.Tensor,
    act: torch.Tensor,
    logp: torch.Tensor,
    adv: torch.Tensor,
    loss_before: float,
    total_steps: int = 15,
    decay: float = 0.8,
) -> tuple[torch.Tensor, int]:
    step_frac = 1.0
    # Get old parameterized policy expression
    theta_old = get_flat_params_from(policy.actor)
    # Change expected objective function gradient = expected_imrpove best this moment
    expected_improve = grads.dot(step_direction)

    final_kl = 0.0

    # While not within_trust_region and not out of total_steps:
    for step in range(total_steps):
        # update theta params
        new_theta = theta_old + step_frac * step_direction
        # set new params as params of net
        set_param_values_to_model(policy.actor, new_theta)

        with torch.no_grad():
            loss = loss_pi(obs, act, logp, adv)
            # compute KL distance between new and old policy
            q_dist = policy.actor(obs)
            # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
        # real loss improve: old policy loss - new policy loss
        loss_improve = loss_before - loss.item()
        logger.log(f'Expected Improvement: {expected_improve} Actual: {loss_improve}')
        if not torch.isfinite(loss):
            logger.log('WARNING: loss_pi not finite')
        elif loss_improve < 0:
            logger.log('INFO: did not improve improve <0')
        elif kl > args.target_kl:
            logger.log('INFO: violated KL constraint.')
        else:
            # step only if surrogate is improved and when within trust reg.
            acceptance_step = step + 1
            logger.log(f'Accept step at i={acceptance_step}')
            final_kl = kl
            break
        step_frac *= decay
    else:
        logger.log('INFO: no suitable step found...')
        step_direction = torch.zeros_like(step_direction)
        acceptance_step = 0

    set_param_values_to_model(policy.actor, theta_old)

    logger.store(
        {
            'Train/KL': final_kl,
        },
    )

    return step_frac * step_direction, acceptance_step

if __name__ == "__main__":
    args = parse_args()

    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # set training steps
    local_steps_per_epoch = args.steps_per_epoch//args.num_envs
    epochs = args.total_steps // args.steps_per_epoch

    # create and wrap the environment
    if args.num_envs > 1:
        env = safety_gymnasium.vector.make(env_id=args.env_id, num_envs=args.num_envs, wrappers=SafeNormalizeObservation)
        env.reset(seed=args.seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        env = SafeNormalizeObservation(env)
    else:
        env = safety_gymnasium.make(args.env_id)
        env.reset(seed=args.seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)

    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
    ).to(device)
    reward_critic_optimizer = torch.optim.Adam(policy.reward_critic.parameters(), lr=args.critic_lr)
    cost_critic_optimizer = torch.optim.Adam(policy.cost_critic.parameters(), lr=args.critic_lr)

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size = args.steps_per_epoch,
        gamma = args.gamma,
        lam = args.lam,
        lam_c = args.lam_c,
        standardized_adv_r=args.standardized_adv_r,
        standardized_adv_c=args.standardized_adv_c,
        device=device,
        num_envs = args.num_envs,
    )

    # set up the logger
    dict_args = vars(args)
    exp_name = "-".join([args.env_id, "trpo", "seed-" + str(args.seed)])
    logger = EpochLogger(
        base_dir=args.log_dir,
        seed=str(args.seed),
        exp_name=exp_name,
        use_tensorboard=args.use_tensorboard,
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")

    start_time = time.time()

    # training loop
    for epoch in range(epochs):
        rollout__start_time = time.time()    
        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        ep_ret, ep_cost, ep_len = np.zeros(args.num_envs), np.zeros(args.num_envs), np.zeros(args.num_envs)

        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = env.step(act.detach().squeeze().cpu().numpy())
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device) for x in (next_obs, reward, cost, terminated, truncated)
            )
            if 'final_observation' in info:
                info['final_observation'] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info['final_observation']
                    ],
                )
                info['final_observation'] = torch.as_tensor(
                    info['final_observation'],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(obs[idx], deterministic=False)
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info['final_observation'][idx],
                                    deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                        **{
                            "Metrics/EpRet": np.mean(rew_deque), 
                            "Metrics/EpCosts": np.mean(cost_deque),
                            "Metrics/EpLen": np.mean(len_deque), 
                          }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r = last_value_r, last_value_c=last_value_c, idx = idx)
        rollout_end_time = time.time()
    
        # update policy
        data = buffer.get()
        fvp_obs = data['obs'][:: args.fvp_sample_freq]
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()

        # compute loss_pi
        distribution = policy.actor(data['obs'])
        log_prob = distribution.log_prob(data['act']).sum(dim=-1)
        ratio = torch.exp(log_prob - data['log_prob'])
        loss_pi = -(ratio * data['adv_r']).mean()

        loss_pi.backward()

        grads = -get_flat_gradients_from(policy.actor)
        x = conjugate_gradients(fvp, policy, fvp_obs, grads, args.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, fvp(x, policy, fvp_obs,))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * args.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        theta_new = theta_old + step_direction
        set_param_values_to_model(policy.actor, theta_new)

        logger.store(
            **{
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                "Loss/Loss_actor": loss_pi.mean().item(),
            },
        )

        dataloader = DataLoader(
            dataset=TensorDataset(
                data['obs'], 
                data['target_value_r'], 
                ),
            batch_size=args.batch_size,
            shuffle=True,
        )
        for _ in track(range(args.update_iters), description='Updating...'):
            for (
                obs_b,
                target_value_r_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                for param in policy.reward_critic.parameters():
                    loss_r += param.pow(2).sum() * args.critic_norm_coef
                loss_r.backward()
                clip_grad_norm_(
                    policy.reward_critic.parameters(),
                    args.max_grad_norm,
                )
                reward_critic_optimizer.step()

                logger.store(**{"Loss/Loss_reward_critic": loss_r.mean().item(),})
        update_end_time = time.time()

        # log data
        logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)
        logger.log_tabular("Metrics/EpCosts", min_and_max=True, std=True)
        logger.log_tabular("Metrics/EpLen", min_and_max=True)
        logger.log_tabular('Train/Epoch', epoch+1)
        logger.log_tabular('Train/TotalSteps', (epoch+1)*args.steps_per_epoch)
        logger.log_tabular('Train/KL')
        logger.log_tabular("Loss/Loss_reward_critic")
        logger.log_tabular("Loss/Loss_actor")
        logger.log_tabular('Time/Rollout', rollout_end_time - rollout__start_time)
        logger.log_tabular('Time/Update', update_end_time - rollout_end_time)
        logger.log_tabular('Value/RewardAdv', data['adv_r'].mean().item())
        logger.log_tabular('Value/CostAdv', data['adv_c'].mean().item())
        logger.log_tabular('Misc/Alpha')
        logger.log_tabular('Misc/FinalStepNorm')
        logger.log_tabular('Misc/xHx')
        logger.log_tabular('Misc/gradient_norm')
        logger.log_tabular('Misc/H_inv_g')

        logger.dump_tabular()
        if epoch % 100 == 0:
            logger.torch_save(itr=epoch)
    logger.close()
