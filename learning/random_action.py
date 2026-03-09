"""Create a LeapXELA environment instance and take random actions."""

import jax
import jax.numpy as jp
from mujoco_playground import registry

from jax import config
config.update("jax_debug_nans", True)

from util import inspect_state_data, check_state_for_nan
def main():
  # Load the LeapXELACubeReorient environment
  env_name = "LeapXELACubeReorient"
  env = registry.manipulation.load(env_name)
  
  print(f"Loaded environment: {env_name}")
  print(f"Action size: {env.action_size}")
  # JIT compile reset and step functions for better performance
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  
  # Initialize random key
  rng = jax.random.PRNGKey(42)
  
  # Reset the environment
  print("Resetting environment...")
  rng, reset_key = jax.random.split(rng)
  state = jit_reset(reset_key)
  inspect_state_data(state)
  check_state_for_nan(state)
  # TODO check state for NaN
  print(f"Environment reset. Episode length: {env._config.episode_length}")
  
  # Take random actions for a few steps
  num_steps = 100
  print(f"\nTaking {num_steps} random actions...")
  
  for step in range(num_steps):
    # Generate random action in range [-1, 1]
    rng, action_key = jax.random.split(rng)
    action = jax.random.uniform(
        action_key, (env.action_size,), minval=-1.0, maxval=1.0
    )
    action_zero = jp.zeros(env.action_size)
    action = action_zero
    
    # Step the environment
    # TODO check state or action for NaN
    state = jit_step(state, action)
    check_state_for_nan(state)
    # Print progress every 10 steps
    if (step + 1) % 10 == 0:
      reward = float(state.reward)
      done = bool(state.done)
      print(f"Step {step + 1}: reward={reward:.4f}, done={done}")
      
      # Reset if episode is done
      if done:
        print("Episode done, resetting...")
        rng, reset_key = jax.random.split(rng)
        state = jit_reset(reset_key)
  
  print("\nFinished taking random actions!")


if __name__ == "__main__":
  main()

