#!/usr/bin/env python3
"""
MARLPSO N-Dimensional Environment
Lightweight MARL-PSO hybrid optimizer supporting n-dimensional optimization problems.
Implements state-based intelligent perturbation mechanisms.
"""

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple, Any, Optional, List, Union
import json
import os
from datetime import datetime

def _handle_bounds(bounds: Union[Tuple[float, float], List[list], np.ndarray], dim: int) -> np.ndarray:
    """
    Unify different bound formats to (2, dim) numpy array.
    - bounds[0, :] is lower bound
    - bounds[1, :] is upper bound
    """
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
        min_b, max_b = bounds
        bounds_arr = np.array([[min_b] * dim, [max_b] * dim], dtype=np.float32)
    elif isinstance(bounds, (list, np.ndarray)):
        bounds_arr = np.array(bounds, dtype=np.float32)
        if bounds_arr.shape == (dim, 2):
            bounds_arr = bounds_arr.T
        elif bounds_arr.shape != (2, dim):
            raise ValueError(f"Provided bounds shape {bounds_arr.shape} is incorrect, expected (2, {dim}) or ({dim}, 2)")
    else:
        raise TypeError(f"Unsupported bounds format: {type(bounds)}")
    
    return bounds_arr

class MARLPSOParticle:
    """N-dimensional MARLPSO Particle - supports state-based intelligent perturbation"""
    
    def __init__(self, bounds: np.ndarray, dim: int, particle_id: str, 
                 perturbation_mode: str = "scheme3", w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.id = particle_id
        self.bounds = bounds  # shape (2, dim)
        self.dim = dim
        self.range = (self.bounds[1] - self.bounds[0]).astype(np.float32)
        self.perturbation_mode = perturbation_mode  # "scheme3" or "pure_pso"
        
        # Initialize position and velocity
        self.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim).astype(np.float32)
        v_max_init = self.range * 0.1
        self.velocity = np.random.uniform(-v_max_init, v_max_init, self.dim).astype(np.float32)
        
        # Individual best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.prev_best_fitness = float('inf')
        
        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = self.range * 0.1
        
        # State tracking
        self.fitness_stagnation = 0
        self.last_best_fitness = float('inf')
        
    def update_classic_pso(self, global_best_pos: np.ndarray, r1: np.ndarray = None, r2: np.ndarray = None) -> np.ndarray:
        """Calculate classic PSO velocity"""
        if r1 is None:
            r1 = np.random.rand(self.dim)
        if r2 is None:
            r2 = np.random.rand(self.dim)

        v_pso = (self.w * self.velocity +
                self.c1 * r1 * (self.best_position - self.position) +
                self.c2 * r2 * (global_best_pos - self.position))
        v_pso = np.clip(v_pso, -self.v_max, self.v_max)

        return v_pso
    
    def apply_rl_action_scheme3(self, v_pso: np.ndarray, action: np.ndarray, _global_best_pos: np.ndarray) -> np.ndarray:
        """State-based intelligent perturbation - continuous action control"""
        # Map [-1,1] to [0,1]
        trigger_thresh = (action[0] + 1) / 2.0
        mode_param = (action[1] + 1) / 2.0

        stagnation_ratio = min(self.fitness_stagnation / 10.0, 1.0)

        if stagnation_ratio > trigger_thresh:
            if mode_param < 0.25:
                # Mode 1: Random Jump
                jump_intensity = 0.05 + (mode_param / 0.25) * 0.25
                perturbation = np.random.uniform(-1, 1, self.dim) * self.range * jump_intensity

            elif mode_param < 0.5:
                # Mode 2: Lévy Flight
                levy_scale = 0.005 + ((mode_param - 0.25) / 0.25) * 0.015
                levy = np.random.standard_cauchy(self.dim) * levy_scale
                levy = np.clip(levy, -3, 3)
                perturbation = levy * self.range * (1 + stagnation_ratio)

            elif mode_param < 0.75:
                # Mode 3: Opposition-based exploration
                reverse_factor = 0.5 + ((mode_param - 0.5) / 0.25) * 1.5
                perturbation = -v_pso * reverse_factor * (1 + stagnation_ratio)

            else:
                # Mode 4: Dimension Shuffle
                if self.dim > 1:
                    shift_param = (mode_param - 0.75) / 0.25
                    shift = max(1, int(shift_param * self.dim))
                    intensity = mode_param
                    perturbation = np.roll(v_pso, shift) * intensity
                else:
                    intensity = mode_param
                    perturbation = v_pso * intensity
        else:
            perturbation = np.zeros(self.dim)

        return v_pso + perturbation
    
    def apply_rl_action(self, v_pso: np.ndarray, action: np.ndarray, global_best_pos: np.ndarray) -> Any:
        """Apply RL Agent action"""
        if self.perturbation_mode == "scheme3":
            return self.apply_rl_action_scheme3(v_pso, action, global_best_pos)
        return None
    
    def move(self, v_real: np.ndarray):
        """Update particle position"""
        self.position = self.position + v_real
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.velocity = v_real.copy()
    
    def update_fitness(self, new_fitness: float):
        """Update fitness and track stagnation"""
        if np.isnan(new_fitness) or np.isinf(new_fitness):
            new_fitness = 1e10
        
        self.new_fitness = new_fitness
        self.prev_best_fitness = self.best_fitness

        if abs(self.last_best_fitness) > 1e-10:
            relative_change = abs(new_fitness - self.last_best_fitness) / abs(self.last_best_fitness)
        else:
            relative_change = abs(new_fitness - self.last_best_fitness)
            
        if relative_change < 1e-2:
            self.fitness_stagnation += 1
        else:
            self.fitness_stagnation = 0

        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.best_position = self.position.copy()

        self.last_best_fitness = new_fitness

class MARLPSOEnvironment(MultiAgentEnv):
    """N-dimensional MARLPSO Multi-Agent Environment"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        self.dim = config.get("dim", 2)
        self.fitness_func = config.get("fitness_func", None)
        raw_bounds = config.get("bounds", (-5.12, 5.12))
        self.f_opt = config.get("f_opt", 0.0)

        self.bounds = _handle_bounds(raw_bounds, self.dim)
        self.perturbation_mode = config.get("perturbation_mode", "scheme3")
        
        if self.fitness_func is None:
            raise ValueError("fitness_func must be provided")
        
        self.num_particles = config.get("num_particles", 20)
        self.max_iterations = config.get("max_iterations", 80)
        self.target_fitness = config.get("target_fitness", 1e-12)
        self.w = config.get("w", 0.7)
        self.c1 = config.get("c1", 1.5)
        self.c2 = config.get("c2", 1.5)
        self.range = self.bounds[1] - self.bounds[0]
        self.range_avg_for_reward = np.mean(self.range)
        
        self.agents = [f"particle_{i}" for i in range(self.num_particles)]
        
        if self.perturbation_mode == "scheme3":
            # 3n+2 dimensions [rel_pos(n), rel_pbest(n), norm_vel(n), fitness_rank(1), stagnation_ratio(1)]
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3*self.dim + 2,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3*self.dim + 1,), dtype=np.float32
            )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.observation_spaces = {agent: self.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.action_space for agent in self.agents}
        
        self.particles = [MARLPSOParticle(self.bounds, self.dim, agent_id, self.perturbation_mode, self.w, self.c1, self.c2) 
                         for agent_id in self.agents]
        
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.prev_global_best_fitness = float('inf')
        
        self.iteration = 0
        self.fitness_history = []
        
        self.function_name = config.get("function_name", "unknown_function")
        self.fitness_data_file = config.get("fitness_data_file", None)
        self.episode_count = 0
        self.current_episode_fitness_history = []
    
    def reset(self, *, seed=None, options=None):
        self.particles = [MARLPSOParticle(self.bounds, self.dim, agent_id, self.perturbation_mode, self.w, self.c1, self.c2)
                          for agent_id in self.agents]
        
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.prev_global_best_fitness = float('inf')

        for particle in self.particles:
            fitness = self.fitness_func(particle.position)
            particle.update_fitness(fitness)
        
        self._update_global_best()
        
        self.iteration = 0
        self.fitness_history = []
        self.episode_count += 1
        self.current_episode_fitness_history = []
        
        observations = self._get_observations()
        infos = {agent: {"dim": self.dim, "perturbation_mode": self.perturbation_mode} for agent in self.agents}
        
        return observations, infos
    
    def step(self, action_dict: Dict[str, np.ndarray]):
        self.iteration += 1
        self.prev_global_best_fitness = self.global_best_fitness

        for i, agent in enumerate(self.agents):
            if agent in action_dict:
                particle = self.particles[i]
                action = action_dict[agent]
                v_pso = particle.update_classic_pso(self.global_best_position)
                control_result = particle.apply_rl_action(v_pso, action, self.global_best_position)

                if isinstance(control_result, np.ndarray):
                    v_real = control_result
                else:
                    v_real = v_pso
                
                particle.move(v_real)
                new_fitness = self.fitness_func(particle.position)
                particle.update_fitness(new_fitness)
        
        self._update_global_best()
        rewards = self._calculate_rewards(action_dict)
        self.fitness_history.append(self.global_best_fitness)
        self.current_episode_fitness_history.append(self.global_best_fitness)
        
        terminated = False
        truncated = (self.iteration >= self.max_iterations)
        
        if (terminated or truncated) and self.fitness_data_file:
            self._save_fitness_data()
        
        observations = self._get_observations()
        infos = {
            agent: {
                "global_best_fitness": self.global_best_fitness,
                "iteration": self.iteration,
                "particle_fitness": self.particles[i].best_fitness,
                "dim": self.dim,
                "perturbation_mode": self.perturbation_mode,
                "stagnation": self.particles[i].fitness_stagnation
            } 
            for i, agent in enumerate(self.agents)
        }
        
        terminateds = {agent: terminated for agent in self.agents}
        terminateds["__all__"] = terminated
        truncateds = {agent: truncated for agent in self.agents}
        truncateds["__all__"] = truncated
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _update_global_best(self):
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        observations = {}
        fitnesses = [p.best_fitness for p in self.particles]
        ranks = np.argsort(np.argsort(fitnesses))
        
        for i, agent in enumerate(self.agents):
            obs = self._get_particle_observation(i, ranks[i])
            observations[agent] = obs
        return observations
    
    def _get_particle_observation(self, particle_idx: int, rank: int) -> np.ndarray:
        particle = self.particles[particle_idx]
        range_with_epsilon = self.range + 1e-8
        if self.global_best_position is None:
            self.global_best_position = particle.position.copy()

        rel_pos = (particle.position - self.global_best_position) / range_with_epsilon
        rel_pbest = (particle.position - particle.best_position) / range_with_epsilon
        norm_velocity = particle.velocity / (range_with_epsilon * 0.1)
        fitness_rank = rank / self.num_particles
        
        if self.perturbation_mode == "scheme3":
            stagnation_ratio = min(particle.fitness_stagnation / 10.0, 1.0)
            observation = np.concatenate([
                rel_pos, rel_pbest, norm_velocity, [fitness_rank], [stagnation_ratio]
            ]).astype(np.float32)
        else:
            observation = np.concatenate([
                rel_pos, rel_pbest, norm_velocity, [fitness_rank]
            ]).astype(np.float32)
        
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -1.0, 1.0)
        return observation
    
    def _calculate_rewards(self, action_dict: Dict) -> Dict[str, float]:
        rewards = {}
        global_improvement = self.prev_global_best_fitness - self.global_best_fitness
        if global_improvement > 0:
            raw_global_reward = global_improvement / self.range_avg_for_reward
            global_reward = np.tanh(raw_global_reward * 10)
        else:
            global_reward = 0.0

        for i, agent in enumerate(self.agents):
            if agent not in action_dict:
                rewards[agent] = 0.0
                continue
            particle = self.particles[i]
            individual_improvement = particle.prev_best_fitness - particle.best_fitness
            if individual_improvement > 0:
                raw_individual_reward = individual_improvement / self.range_avg_for_reward
                individual_reward = np.tanh(raw_individual_reward * 10)
            else:
                individual_reward = -0.01
            rewards[agent] = 0.2 * individual_reward + 0.8 * global_reward
        return rewards
    
    def _save_fitness_data(self):
        if not self.fitness_data_file:
            return
        try:
            episode_data = {
                "episode": self.episode_count,
                "function": self.function_name,
                "dim": self.dim,
                "perturbation_mode": self.perturbation_mode,
                "final_fitness": float(self.global_best_fitness),
                "iterations": self.iteration,
                "fitness_history": [float(f) for f in self.current_episode_fitness_history],
                "timestamp": datetime.now().isoformat(),
                "theoretical_optimum": float(self.f_opt)
            }
            dir_name = os.path.dirname(self.fitness_data_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            existing_data = []
            if os.path.exists(self.fitness_data_file):
                with open(self.fitness_data_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except:
                        existing_data = []
            existing_data.append(episode_data)
            with open(self.fitness_data_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except:
            pass
