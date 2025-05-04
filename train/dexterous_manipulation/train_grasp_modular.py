import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

import cv2

# Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils

import jax
from jax import numpy as jp

import matplotlib
matplotlib.use("TkAgg")  # Or "QtAgg" / "Agg" if needed
# sudo apt install python3-tk

from matplotlib import pyplot as plt

# import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

# Import The Playground
from mujoco_playground import wrapper
from mujoco_playground import registry

from mujoco_playground.config import manipulation_params


print(f"list of the whole manipulation envs::\n {registry.manipulation.ALL}\n")

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

n_episodes = 1
env_name = 'LeapGrasp'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
print(f"env_cfg:: {env_cfg}\n")

ppo_params = manipulation_params.brax_ppo_config(env_name)
print(f"\nppo_params:: {ppo_params}\n")

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.clf()  # Clear the figure instead of clearing output
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

    plt.draw()  # Redraw the plot
    plt.pause(0.1)  # Allow GUI event loop to update


def augment_state(state,finger):
  """
  Takes state and current finger that needs to be controlled
  and outputs augmented state
  """
  pass

def main():

  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

  # I need to change PPO code so that it
  # augment state and in training it calls the network to get action per finger
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory = network_factory,
      progress_fn = progress,
      seed = 1
  )

  make_inference_fn, params, metrics = train_fn(
      environment = env,
      wrap_env_fn = wrapper.wrap_for_brax_training,
  )

  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  rng = jax.random.PRNGKey(42)

  # Initialize rollout list
  rollout = []

  # Run multiple episodes
  for _ in range(n_episodes):
      state = jit_reset(rng)
      episode_rollout = [state]  # Store episode states separately

      # TODO: Modify this code so that it
      # augment the state with current finger being controlled
      # feed that to the network, then remove the augmentation and feed it back
      # to the step function
      for _ in range(env_cfg.episode_length):
          act_rng, rng = jax.random.split(rng)
          fingers_commands = []
          # call the network to get action per finger
          for finger in ["FF","MF","RF","TH"]:
            augmented_state = augment_state(state.obs)
            finger_ctrl, _ = jit_inference_fn(augmented_state, act_rng)
            fingers_commands.append(finger_ctrl)

          ctrl = jp.concatenate(fingers_commands)
          state = jit_step(state.obs, ctrl)
          episode_rollout.append(state)

      rollout.extend(episode_rollout)  # Add to the main rollout list

  # Rendering settings
  render_every = 1
  frames = env.render(rollout[::render_every])
  rewards = [s.reward for s in rollout]



  def display_video(frames, fps=30):
      for frame in frames:
          cv2.imshow("Simulation", frame)
          if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Press 'q' to exit
              break
      cv2.destroyAllWindows()

  display_video(frames, fps=1.0 / env.dt / render_every)


  # save video
  video_filename = "leap_grasp.mp4"
  fps = 1.0 / env.dt / render_every  # Frame rate based on the environment's timestep
  frame_size = (frames[0].shape[1], frames[0].shape[0])  # Width, Height

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
  video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

  # Write each frame to the video
  for frame in frames:
      frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)
      video_writer.write(frame_bgr)  # Write frame to video

  # Release the VideoWriter
  video_writer.release()
  print(f"Video saved as {video_filename}")













if __name__ == "__main__":
  main()

