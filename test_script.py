#!/usr/bin/env python3
"""
Simple test script to verify MARLPSO environment and basic functionality.
"""

import numpy as np
from marlpso.env import MARLPSOEnvironment
from marlpso.functions import get_cec_function

def test_environment():
    print("Testing MARLPSO Environment...")
    
    # 1. Setup a Simple CEC function (e.g., CEC2017 F1)
    dim = 10
    func = get_cec_function(2017, 1, dim=dim)
    
    # 2. Configure Environment
    config = {
        "dim": dim,
        "fitness_func": func.evaluate,
        "bounds": func.bounds,
        "num_particles": 5,
        "max_iterations": 10,
        "perturbation_mode": "scheme3"
    }
    
    # 3. Initialize Environment
    env = MARLPSOEnvironment(config)
    obs, info = env.reset()
    
    print(f"Env initialized with {len(env.agents)} particles.")
    print(f"Observation shape: {obs[env.agents[0]].shape}")
    
    # 4. Perform a single step
    actions = {agent: env.action_space.sample() for agent in env.agents}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    
    print("Step successful.")
    print(f"Current best fitness: {env.global_best_fitness:.6f}")
    
    print("\nMARLPSO basic test passed!")

if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
