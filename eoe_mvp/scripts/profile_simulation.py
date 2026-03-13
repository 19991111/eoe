#!/usr/bin/env python3
"""Profiling script to find GPU utilization bottleneck"""

import sys
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents

device = 'cuda:0'
n_agents = 1000
n_steps = 100

# Create environment and agents
env = EnvironmentGPU(width=100, height=100, resolution=1.0, device=device)
agents = BatchedAgents(n_agents=n_agents, env_width=100, env_height=100, device=device, init_energy=150.0)

# Random brain weights
max_nodes = 16
brain_weights = torch.randn(n_agents, max_nodes, max_nodes, device=device) * 0.5

# Timing containers
timing = {
    'get_sensors': [],
    'neural_forward': [],
    'agent_step': [],
    'env_step': [],
    'total': []
}

print("Profiling 100 steps with 1000 agents on", device)
print("-" * 50)

for step in range(n_steps):
    step_start = time.perf_counter()
    
    # 1. Get sensors
    t0 = time.perf_counter()
    sensors = agents.get_sensors(env)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    timing['get_sensors'].append(time.perf_counter() - t0)
    
    # 2. Neural forward
    t0 = time.perf_counter()
    W1 = brain_weights[:, :sensors.shape[1], :16]
    hidden = torch.bmm(sensors.unsqueeze(1), W1).squeeze(1)
    hidden = F.relu(hidden)
    W2 = brain_weights[:, :16, :5]
    brain_outputs = torch.bmm(hidden.unsqueeze(1), W2).squeeze(1)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    timing['neural_forward'].append(time.perf_counter() - t0)
    
    # 3. Agent step
    t0 = time.perf_counter()
    permeabilities = torch.sigmoid(brain_outputs[:, 0])
    thrust_x = torch.tanh(brain_outputs[:, 1])
    thrust_y = torch.tanh(brain_outputs[:, 2])
    signals = torch.relu(brain_outputs[:, 3])
    defenses = torch.sigmoid(brain_outputs[:, 4])
    
    new_velocities = torch.stack([thrust_x, thrust_y], dim=1) * permeabilities.unsqueeze(1) * 5.0
    new_positions = agents.state.positions + new_velocities * 1.0
    new_positions[:, 0] = new_positions[:, 0] % 100.0
    new_positions[:, 1] = new_positions[:, 1] % 100.0
    new_thetas = torch.atan2(new_velocities[:, 1], new_velocities[:, 0])
    energy_cost = (permeabilities * 0.01 + torch.sum(torch.abs(new_velocities), dim=1) * 0.01 + signals * 0.1)
    new_energies = agents.state.energies - energy_cost * 1.0
    new_is_alive = new_energies > 0
    
    agents.state.positions = new_positions
    agents.state.velocities = new_velocities
    agents.state.energies = new_energies
    agents.state.thetas = new_thetas
    agents.state.permeabilities = permeabilities
    agents.state.defenses = defenses
    agents.state.signals = signals
    agents.state.is_alive = new_is_alive
    torch.cuda.synchronize() if device.startswith('cuda') else None
    timing['agent_step'].append(time.perf_counter() - t0)
    
    # 4. Environment step
    t0 = time.perf_counter()
    env.step()
    torch.cuda.synchronize() if device.startswith('cuda') else None
    timing['env_step'].append(time.perf_counter() - t0)
    
    timing['total'].append(time.perf_counter() - step_start)
    
    if (step + 1) % 20 == 0:
        print(f"Step {step+1}: {timing['total'][-1]*1000:.2f}ms")

print("-" * 50)
print("Average times:")
for k, v in timing.items():
    avg_ms = sum(v) / len(v) * 1000
    pct = avg_ms / (sum(timing['total']) / len(timing['total'])) * 100
    print(f"  {k:15s}: {avg_ms:6.2f}ms ({pct:5.1f}%)")