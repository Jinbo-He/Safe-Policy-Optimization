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
import torch
import torch.nn as nn

from safepo.utils.distributions import DiagGaussian, Categorical


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        # self.action_type = action_space.__class__.__name__
        # action_dim = action_space.shape[0]
        # self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
        if action_space.__class__.__name__ == "tuple":
            self.hybird_action = True
            action_dims = [action_space[0].n, action_space[1].shape[0]]
            action_outs = []
            self.action_outs = nn.ModuleList([Categorical(inputs_dim, action_dims[0], use_orthogonal, gain),
                                              DiagGaussian(inputs_dim, action_dims[1], use_orthogonal, gain, args),])
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        
        if self.hybird_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)
            # actions[0] = torch.argmax(actions[0], dim=-1, keepdim=True)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.hybird_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            """-----------------Discrete_act------------------"""
            action_distribution = self.action_outs[0](x)
            action_log_probs.append(action_distribution.log_probs(action[0].unsqueeze(-1)))
            if active_masks is not None:
                dist_entropy.append((action_distribution.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
            else:
                dist_entropy.append(action_distribution.entropy() / action_log_probs[-1].size(0))
            
            """-----------------continuous_act------------------"""
            action_distribution = self.action_outs[1](x)
            action_log_probs.append(action_distribution.log_probs(action[1:].transpose(0, 1)))
            
            if active_masks is not None:
                dist_entropy.append((action_distribution.entropy() * active_masks).sum() / active_masks.sum())
            else:
                dist_entropy.append(action_distribution.entropy().mean())
            
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] + dist_entropy[1]
            # dist_entropy = sum(dist_entropy)
            
            return action_log_probs, dist_entropy
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_logits = self.action_out(x, available_actions)
        action_mu = action_logits.mean
        action_std = action_logits.stddev
        action_log_probs = action_logits.log_probs(action)
        all_probs = None
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
