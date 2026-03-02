#!/usr/bin/env python3
"""
MARLPSO Training Script
Supports training MARLPSO models on CEC benchmark functions using Ray RLlib.
"""

import os
import argparse
import json
import csv
import time
import ast
from typing import Dict, Any, Callable, Tuple, Union, List
from datetime import datetime
import numpy as np

from marlpso.env import MARLPSOEnvironment, _handle_bounds
from marlpso.functions import get_cec_function, get_cec_info

# Try to import Ray
try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    RAY_AVAILABLE = True
except ImportError:
    print("Warning: Ray not installed, training cannot proceed.")
    RAY_AVAILABLE = False

def create_training_config(fitness_func: Callable,
                            function_name: str,
                            dim: int,
                            bounds: Union[Tuple[float, float], np.ndarray],
                            f_opt: float = 0.0,
                            num_particles: int = 20,
                            max_iterations: int = 100,
                            perturbation_mode: str = "scheme1",
                            fitness_data_file: str = "default.json",
                            w: float = 0.7,
                            c1: float = 1.5,
                            c2: float = 1.5,
                            lr: float = 3e-6,
                            gamma: float = 0.999,
                            lambda_: float = 0.999,
                            train_batch_size: int = 160,
                            minibatch_size: int = 160,
                            clip_param: float = 0.3,
                            entropy_coeff_schedule: list = None,
                            fcnet_hiddens: list = None
                            ) -> 'PPOConfig':
    """Create RLlib training configuration"""
    
    env_config = {
        "dim": dim,
        "function_name": function_name,
        "fitness_func": fitness_func,
        "bounds": bounds,
        "f_opt": f_opt,
        "num_particles": num_particles,
        "max_iterations": max_iterations,
        "target_fitness": 0.01,
        "perturbation_mode": perturbation_mode,
        "fitness_data_file": fitness_data_file,
        "w": w,
        "c1": c1,
        "c2": c2
    }
    
    config = (
        PPOConfig()
        .environment(
            env=MARLPSOEnvironment,
            env_config=env_config
        )
        .multi_agent(
            policies={
                "pso_policy": PolicySpec(
                    policy_class=None,
                    config={}
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "pso_policy",
            policies_to_train=["pso_policy"]
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=lambda_,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            clip_param=clip_param,
            entropy_coeff_schedule=entropy_coeff_schedule if entropy_coeff_schedule is not None else [(0, 0.01), (60, 0.01)],
            model={
                "fcnet_hiddens": fcnet_hiddens if fcnet_hiddens is not None else [2560, 512, 512],
            }
        )
        .env_runners(
            num_env_runners=0,
            explore=True
        )
        .resources(
            num_gpus=0
        )
        .debugging(
            log_level="INFO"
        )
    )
    
    return config

def parse_function_name(function_name: str, dim: int) -> Tuple[object, str]:
    """
    Parse CEC function name and return function instance.
    Format example: cec2017_f5
    """
    function_name = function_name.lower().strip()
    
    if function_name.startswith("cec") and "_f" in function_name:
        try:
            parts = function_name.split("_f")
            year_part = parts[0]
            year = int(year_part[3:])
            func_id = int(parts[1])
            
            func = get_cec_function(year, func_id, dim)
            description = f"CEC{year} F{func_id} ({dim}D)"
            return func, description
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid CEC function format: {function_name}. "
                           f"Use format like: cec2017_f5. Error: {e}")
    else:
        raise ValueError(f"Only CEC functions are supported (e.g., cec2017_f5). Got: {function_name}")

def train_marlpso(function_name: str,
                   dim: int,
                   training_iterations: int = 100,
                   num_particles: int = 20,
                   max_pso_iterations: int = 10,
                   w: float = 0.7,
                   c1: float = 1.5,
                   c2: float = 1.5,
                   perturbation_mode: str = "scheme3",
                   checkpoint_dir: str = "./checkpoints",
                   bounds: Union[Tuple[float, float], np.ndarray, List] = None,
                   verbose: bool = True,
                   fitness_data_file: str = "default.json",
                   lr: float = 3e-6,
                   gamma: float = 0.999,
                   lambda_: float = 0.999,
                   train_batch_size: int = 160,
                   minibatch_size: int = 160,
                   clip_param: float = 0.3,
                   entropy_coeff_schedule: list = None,
                   fcnet_hiddens: list = None) -> Dict[str, Any]:
    """Main training loop for MARLPSO"""
    
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not installed, cannot train.")
    
    try:
        func, func_description = parse_function_name(function_name, dim)
        fitness_func = func.evaluate
        func_bounds = bounds if bounds is not None else func.bounds
        func_bounds = _handle_bounds(func_bounds, dim)
        f_opt = func.global_minimum
    except Exception as e:
        print(f"Function parsing failed: {e}")
        raise
    
    experiment_name = f"{function_name}_{dim}d_{perturbation_mode}"
    print(f"=== Training MARLPSO ===")
    print(f"Function: {func_description}")
    print(f"Dimensions: {dim}")
    print(f"Particles: {num_particles}")
    print(f"Training Iterations: {training_iterations}")
    print()
    
    if not ray.is_initialized():
        ray.init(local_mode=False, ignore_reinit_error=True)
    
    config = create_training_config(
        fitness_func=fitness_func,
        function_name=function_name,
        dim=dim,
        bounds=func_bounds,
        f_opt=f_opt,
        num_particles=num_particles,
        max_iterations=max_pso_iterations,
        perturbation_mode=perturbation_mode,
        fitness_data_file=fitness_data_file,
        w=w,
        c1=c1,
        c2=c2,
        lr=lr,
        gamma=gamma,
        lambda_=lambda_,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        clip_param=clip_param,
        entropy_coeff_schedule=entropy_coeff_schedule,
        fcnet_hiddens=fcnet_hiddens
    )
    
    print("Building algorithm...")
    algo = config.build()
    
    checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, experiment_name))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    training_results = []
    best_reward = -float('inf')
    start_time = time.time()
    
    try:
        for i in range(training_iterations):
            result = algo.train()
            
            episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", result.get("episode_reward_mean", 0.0))
            episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", result.get("episode_len_mean", 0.0))
            
            training_results.append({
                "iteration": i + 1,
                "reward": episode_reward_mean
            })
            
            if verbose:
                print(f"Iter {i+1:3d}: Mean Reward = {episode_reward_mean:8.4f}, Mean Length = {episode_len_mean:6.1f}")
            
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                best_checkpoint = os.path.join(checkpoint_dir, "best_model")
                algo.save(best_checkpoint)

        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.1f}s. Best reward: {best_reward:.4f}")
        return {"best_reward": best_reward, "results": training_results}
        
    finally:
        algo.stop()
        if ray.is_initialized():
            ray.shutdown()

def main():
    parser = argparse.ArgumentParser(description="MARLPSO Training with CEC Benchmark Functions")
    parser.add_argument("--function", type=str, default="cec2017_f1", help="CEC function name (e.g., cec2017_f1)")
    parser.add_argument("--dim", type=int, default=10, help="Problem dimension")
    parser.add_argument("--particles", type=int, default=20, help="Number of particles")
    parser.add_argument("--pso_iterations", type=int, default=30, help="PSO iterations per RL step")
    parser.add_argument("--train_iterations", type=int, default=10, help="Total training episodes")
    parser.add_argument("--perturbation_mode", type=str, default="scheme3", choices=["scheme3", "pure_pso"])
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    args = parser.parse_args()
    
    train_marlpso(
        function_name=args.function,
        dim=args.dim,
        training_iterations=args.train_iterations,
        num_particles=args.particles,
        max_pso_iterations=args.pso_iterations,
        perturbation_mode=args.perturbation_mode,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints")
    )

if __name__ == "__main__":
    main()
