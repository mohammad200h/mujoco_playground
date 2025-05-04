# Copyright 2025 DeepMind Technologies Limited
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
"""Grasp task for leap hand."""

from typing import Any, Dict, Optional, Union

# this import is used for debuging
from jax import device_get

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.leap_grasp import base as leap_hand_base
from mujoco_playground._src.manipulation.leap_grasp import leap_grasp_constants as consts


from enum import Enum

class RewardType(Enum):
    JOINT_VEL_JOINT_TORQUE = "joint_vel_joint_torque"
    CUBE_VEL_JOINT_TORQUE = "cube_vel_joint_torque"
    JOINT_VEL_JOINT_TORQUE_DISTANCE_DEPENDENT = "joint_vel_joint_torque_distance_dependent"
    HAND_MAINTAINING_PREGRASP_POSE = "hand_maintaining_pregrasp_pose"


# TODO: Replace this so it's loaded from xml key
cube_initial_location = [-0.05, 0, 0.15]

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.1,
      history_len=1,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              cube_vel_joint_torque= 1,
              termination=-100.0,
              joint_vel_joint_torque = 1,
              joint_vel_joint_torque_distance_dependent = 1,
              hand_maintaining_pregrasp_pose = 1
          ),
          success_reward=100.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
  )


class CubeGrasp(leap_hand_base.LeapHandEnv):
  """Reorient a cube to match a goal orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      reward_type:RewardType = RewardType.HAND_MAINTAINING_PREGRASP_POSE
  ):
    super().__init__(
      xml_path=consts.CUBE_XML.as_posix(),
      config=config,
      config_overrides=config_overrides,
    )
    # Setting reward mode
    self.reward_type = reward_type

    self._post_init()

  def _post_init(self) -> None:
    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)

    self._mocap_quat_close_enough = jp.array( self._mj_model.keyframe("close_enough").mquat)
    self._mocap_quat_far = jp.array( self._mj_model.keyframe("far").mquat)
    self._mocap_pos_close_enough = jp.array( self._mj_model.keyframe("close_enough").mpos)
    self._mocap_pos_far = jp.array( self._mj_model.keyframe("far").mpos)

    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self.pregrasp_pose = jp.array(self._mj_model.keyframe("pregrasp").qpos[self._hand_qids])

    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id
    self._cube_body_id = self._mj_model.body("cube").id
    self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
    self._default_pose = self._init_q[self._hand_qids]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize the goal orientation.
    # rng, goal_rng = jax.random.split(rng)
    # goal_quat = leap_hand_base.uniform_quat(goal_rng)

    # Randomize the hand pose.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize the cube pose.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array(cube_initial_location) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    start_quat = leap_hand_base.uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    data = mjx_env.init(
        self.mjx_model,
        qpos=qpos,
        ctrl=q_hand,
        qvel=qvel,
        mocap_quat=self._mocap_quat_far,
        mocap_pos = self._mocap_pos_far

    )

    rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
    pert_wait_steps = jax.random.randint(
        pert1,
        (1,),
        minval=self._config.pert_config.pert_wait_steps[0],
        maxval=self._config.pert_config.pert_wait_steps[1],
    )
    pert_duration_steps = jax.random.randint(
        pert2,
        (1,),
        minval=self._config.pert_config.pert_duration_steps[0],
        maxval=self._config.pert_config.pert_duration_steps[1],
    )
    pert_lin = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.linear_velocity_pert[0],
        maxval=self._config.pert_config.linear_velocity_pert[1],
    )
    pert_ang = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.angular_velocity_pert[0],
        maxval=self._config.pert_config.angular_velocity_pert[1],
    )
    pert_velocity = jp.array([pert_lin] * 3 + [pert_ang] * 3)

    info = {
        "rng": rng,
        "step": 0,
        "episode_step":0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "qpos_error_history": jp.zeros(self._config.history_len * 16),
        "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
        "goal_quat_dquat": jp.zeros(3),
        # Perturbation.
        "pert_wait_steps": pert_wait_steps,
        "pert_duration_steps": pert_duration_steps,
        "pert_vel": pert_velocity,
        "pert_dir": jp.zeros(6, dtype=float),
        "last_pert_step": jp.array([-jp.inf], dtype=float),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def get_random_action(self,rng):
    # generate a random action

    # Split the RNG key to ensure reproducibility
    rng, subkey = jax.random.split(rng)

    # Generate uniform random values in the range [0, 1) with the same shape as the control ranges
    random_values = jax.random.uniform(subkey, shape=self._lowers.shape)

    # Scale and shift the random values to fit within the actuator control ranges
    actions = self._lowers + random_values * (self._uppers - self._lowers)

    return actions, rng

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Apply external force to cube driving it toward palm of the hand
    if not self.impulse_applied:
      data = state.data.replace(qfrc_applied= self.get_impulse())
      state = state.replace(data=data)
    else:
      data = state.data.replace(qfrc_applied= self.head_toward_the_hand(state.data))
      state = state.replace(data=data)


    mocap_quat = jp.where(
      self.cube_is_close_enough(data),
      self._mocap_quat_close_enough,
      state.data.mocap_quat
    )
    data = state.data.replace(mocap_quat= mocap_quat)
    state = state.replace(data=data)
    mocap_pos = jp.where(
      self.cube_is_close_enough(data),
      self._mocap_pos_close_enough,
      state.data.mocap_pos
    )
    data = state.data.replace(mocap_pos= mocap_pos)
    state = state.replace(data=data)

    # Apply control and step the physics.
    delta = action * self._config.action_scale
    motor_targets = state.data.ctrl + delta
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    motor_targets = (
        self._config.ema_alpha * motor_targets
        + (1 - self._config.ema_alpha) * state.info["motor_targets"]
    )

    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    state.info["motor_targets"] = motor_targets

    # Power Grasp should reduce acceleration and velocity of cube to zero
    # It should also have contact with Thumb and one other finger
    # success = (self._cube_linear_vel_is_zero(data) +
    #            self._cube_angular_vel_is_zero(data) +
    #            self._there_is_a_contact_with_thumb(data) +
    #            self._there_is_a_contact_with_any_of_ff_mf_rf(data)
    # )
    # success = 0.0

    # state.info["steps_since_last_success"] = jp.where(
    #     success, 0, state.info["steps_since_last_success"] + 1
    # )
    # state.info["success_count"] = jp.where(
    #     success, state.info["success_count"] + 1, state.info["success_count"]
    # )
    state.metrics["steps_since_last_success"] = state.info[
        "steps_since_last_success"
    ]
    state.metrics["success_count"] = state.info["success_count"]

    # TODO: I need to write termination, Maybe just keep the nan condition
    done = self._get_termination(data, state.info)
    obs = self._get_obs(data, state.info)

    # TODO: I need to write reward function
    rewards = self._get_reward(data, action, state.info, state.metrics, done)

    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name


    # reset cube and hand where success
    state.info["rng"], reset_rng = jax.random.split(state.info["rng"])
    new_qpos,new_qvel = self.generate_new_pose_and_velocity(reset_rng)

    # qpos = jp.where(success, new_qpos, data.qpos)
    # qvel = jp.where(success, new_qvel, data.qvel)
    # data = data.replace(qpos=jp.array(qpos))
    # data = data.replace(qvel=jp.array(qvel))
    # state.metrics["reward/success"] = success.astype(float)
    # reward += success * self._config.reward_config.success_reward

    # Update info and metrics.
    state.info["step"] += 1
    state.info["episode_step"] += 1

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)

    return state
  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.

    nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    return nans

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    # Hand joint angles.
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    # Joint position error history.
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 16)
        .at[:16]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    def _get_cube_pose(data: mjx.Data) -> jax.Array:
      """Returns (potentially) noisy cube pose (xyz,wxyz)."""
      cube_pos = self.get_cube_position(data)
      cube_quat = self.get_cube_orientation(data)
      info["rng"], pos_rng, ori_rng = jax.random.split(info["rng"], 3)
      noisy_cube_quat = mjx._src.math.normalize(
          cube_quat
          + jax.random.normal(ori_rng, shape=(4,))
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_ori
      )
      noisy_cube_pos = (
          cube_pos
          + (2 * jax.random.uniform(pos_rng, shape=cube_pos.shape) - 1)
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_pos
      )
      return jp.concatenate([noisy_cube_pos, noisy_cube_quat])

    # Noisy cube pose.
    noisy_pose = _get_cube_pose(data)
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    rand_quat = leap_hand_base.uniform_quat(key1)
    rand_pos = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
    rand_pose = jp.concatenate([rand_pos, rand_quat])
    m = self._config.obs_noise.level * jax.random.bernoulli(
        key3, self._config.obs_noise.random_ori_injection_prob
    )
    noisy_pose = noisy_pose * (1 - m) + rand_pose * m

    # Cube position error history.
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - noisy_pose[:3]
    # TODO: cube_pos_error_history need to be renamed to cube_pos_dist_history
    cube_pos_error_history = (
        jp.roll(info["cube_pos_error_history"], 3).at[:3].set(cube_pos_error)
    )
    info["cube_pos_error_history"] = cube_pos_error_history

    # Uncorrupted cube pose for critic.
    cube_pos_error_uncorrupted = palm_pos - self.get_cube_position(data)

    state = jp.concatenate([
        noisy_joint_angles,  # 16
        qpos_error_history,  # 16 * history_len
        cube_pos_error_history,  # 3 * history_len
        info["last_act"],  # 16
    ])

    privileged_state = jp.concatenate([
        state,
        data.qpos[self._hand_qids],
        data.qvel[self._hand_dqids],
        self.get_fingertip_positions(data),
        cube_pos_error_uncorrupted,
        self.get_cube_linvel(data),
        self.get_cube_angvel(data),
        info["pert_dir"],
        data.xfrc_applied[self._cube_body_id],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  # Reward terms.

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics, info, action  # Unused.

    if self.reward_type == RewardType.JOINT_VEL_JOINT_TORQUE:
      rewards = {
         RewardType.JOINT_VEL_JOINT_TORQUE.value :self.joint_vel_joint_torque_reward(data),
      }
    elif self.reward_type == RewardType.CUBE_VEL_JOINT_TORQUE:
      rewards = {
        RewardType.CUBE_VEL_JOINT_TORQUE.value: self.cube_vel_joint_torque_reward(data),
      }
    elif self.reward_type == RewardType.JOINT_VEL_JOINT_TORQUE_DISTANCE_DEPENDENT:
      rewards = {
        RewardType.JOINT_VEL_JOINT_TORQUE_DISTANCE_DEPENDENT.value:self.joint_vel_joint_torque_distance_dependent_reward(data),
      }
    elif self.reward_type == RewardType.HAND_MAINTAINING_PREGRASP_POSE:
      rewards = {
        RewardType.HAND_MAINTAINING_PREGRASP_POSE.value:self.hand_maintaining_pregrasp_pose_reward(data),
      }

    return rewards

  def hand_maintaining_pregrasp_pose_reward_base(self,data:mjx.Data, k: float):
    hand_pose = data.qpos[self._hand_qids]
    diff = self.pregrasp_pose - hand_pose
    return -k * jp.abs(jp.sum(diff))

  def hand_maintaining_pregrasp_pose_reward(self,data:mjx.Data):
    k = self._config.reward_config.scales[RewardType.HAND_MAINTAINING_PREGRASP_POSE.value]
    reward = self.hand_maintaining_pregrasp_pose_reward_base(data, k)

    return reward

  def joint_vel_joint_torque_distance_dependent_reward_base(self,data:mjx.Data, k: float):
    # exp(-k*dq)* | tau| ^2
    joint_torques = self.get_hand_joint_torque(data)

    joint_torque = jp.linalg.norm(joint_torques)
    dq = self.get_hand_joint_velocity(data)

    return jp.mean(jp.exp(-k * dq**2) * joint_torque**2)

  def joint_vel_joint_torque_distance_dependent_reward(self, data: mjx.Data):
    k = self._config.reward_config.scales[RewardType.JOINT_VEL_JOINT_TORQUE_DISTANCE_DEPENDENT.value]
    reward = self.joint_vel_joint_torque_distance_dependent_reward_base(data, k)

    return jp.where(self.cube_is_close_enough(data), reward, 0.0)

  def cube_is_close_enough(self,data: mjx.Data):
    dist =  self.get_palm_position(data) - self.get_cube_position(data)
    return jp.linalg.norm(dist) < 0.03

  def joint_vel_joint_torque_reward_base(self,data:mjx.Data, k:float):
    # exp(-k * dq^2) * joint_torque^2
    joint_torques = self.get_hand_joint_torque(data)

    dq = self.get_hand_joint_velocity(data)

    return jp.mean(jp.exp(-k * dq**2) * joint_torques**2)

  def joint_vel_joint_torque_reward(self, data:mjx.Data):
    # joint velocity is low and torque is high due to applying force onto object

    k = self._config.reward_config.scales[RewardType.JOINT_VEL_JOINT_TORQUE.value]

    return self.joint_vel_joint_torque_reward_base(data, k)

  def cube_vel_joint_torque_reward_base(self, data:mjx.Data, k:float ):
    # exp(-k*(object_palm_velocity)^2) * joint_torque^2
    cube_velocity = self.get_cube_linvel(data)
    joint_torque = self.get_hand_joint_torque(data)

    k = self._config.reward_config.scales[RewardType.CUBE_VEL_JOINT_TORQUE.value]
    v = jp.linalg.norm(cube_velocity)

    return jp.mean(jp.exp(-k * v**2) * joint_torque**2)

  def cube_vel_joint_torque_reward(self, data:mjx.Data):
    # Cube velocity is high since it has not been grasped
    # Hand torque and joint velocity is high since it can move freely
    # since its not applying any force anywhere

    k = self._config.reward_config.scales["cube_vel_joint_torque"]

    return self.cube_vel_joint_torque_reward_base(data, k)

  def _cube_orientation_error(self, data: mjx.Data):
    cube_ori = self.get_cube_orientation(data)
    cube_goal_ori = self.get_cube_goal_orientation(data)
    quat_diff = math.quat_mul(cube_ori, math.quat_inv(cube_goal_ori))
    quat_diff = math.normalize(quat_diff)

    return 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), a_max=1.0))

  def _cube_linear_vel_is_zero(self,data: mjx.Data):
    velocity_vector =  self.get_cube_linvel(data)
    s = jp.sum(velocity_vector)
    return s == 0

  def generate_new_pose_and_velocity(self,rng):
    # Randomize the hand pose.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize the cube pose.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array(cube_initial_location) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )

    start_quat = leap_hand_base.uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    return qpos, qvel

  def _cube_angular_vel_is_zero(self, data:mjx.Data):
    velocity_vector =  self.get_cube_angvel(data)
    s = jp.sum(velocity_vector)
    return s == 0

  def _there_is_a_contact_with_thumb(self, data:mjx.Data):
    s = jp.sum(self.there_is_contact_between_th_and_object(data))
    return s > 0

  def _there_is_a_contact_with_any_of_ff_mf_rf(self,data:mjx.Data):
    total_contact = jp.concatenate([ self.there_is_contact_for_ff_mf_rf("if", data),
                                     self.there_is_contact_for_ff_mf_rf("mf", data),
                                     self.there_is_contact_for_ff_mf_rf("rf", data)
                                  ])

    return jp.sum(total_contact) > 0


  def opposing_force_reward(self, data:mjx.Data):
    sum = self._there_is_a_contact_with_thumb(data) + self._there_is_a_contact_with_any_of_ff_mf_rf(data)

    return  (sum/2 == 1).astype(jp.float32)


  def cube_velocity_is_zero(self, data:mjx.Data):
    sum = self._cube_linear_vel_is_zero(data) + self._cube_angular_vel_is_zero(data)

    return (sum/2 == 1).astype(jp.float32)
  # Perturbation.

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      directory = jax.random.normal(rng, (6,))
      return directory / jp.linalg.norm(directory)

    def get_xfrc(
        state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
      force = (
          u_t
          * self._cube_mass
          * state.info["pert_vel"]
          / (state.info["pert_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
      return xfrc_applied

    step, last_pert_step = state.info["step"], state.info["last_pert_step"]
    start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
    start_pert &= step != 0  # No perturbation at the beginning of the episode.
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert_interval = duration < state.info["pert_duration_steps"]

    pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
    xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

    state.info["pert_dir"] = pert_dir
    state.info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc)
    return state.replace(data=data)

def domain_randomize(model: mjx.Model, rng: jax.Array):
  mj_model = CubeGrasp().mj_model
  cube_geom_id = mj_model.geom("cube").id
  cube_body_id = mj_model.body("cube").id
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
  hand_body_names = [
      "palm",
      "if_bs",
      "if_px",
      "if_md",
      "if_ds",
      "mf_bs",
      "mf_px",
      "mf_md",
      "mf_ds",
      "rf_bs",
      "rf_px",
      "rf_md",
      "rf_ds",
      "th_mp",
      "th_bs",
      "th_px",
      "th_ds",
  ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

  @jax.vmap
  def rand(rng):
    # Cube friction: =U(0.1, 0.5).
    rng, key = jax.random.split(rng)
    cube_friction = jax.random.uniform(key, (1,), minval=0.1, maxval=0.5)
    geom_friction = model.geom_friction.at[
        cube_geom_id : cube_geom_id + 1, 0
    ].set(cube_friction)

    # Fingertip friction: =U(0.5, 1.0).
    fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
    geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
        fingertip_friction
    )

    # Scale cube mass: *U(0.8, 1.2).
    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
    cube_mass = model.body_mass[cube_body_id]
    body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass
    )
    dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(16,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(16,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    # Joint stiffness: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.8, maxval=1.2
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Joint damping: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key, (16,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes
