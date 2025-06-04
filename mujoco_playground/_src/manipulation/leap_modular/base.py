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
"""Base classes for leap hand."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_modular import leap_hand_constants as consts
from enum import Enum
import numpy as np



def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "leap_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class LeapHandEnv(mjx_env.MjxEnv):
  """Base class for LEAP hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  # Sensor readings.

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angacc")

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_upvector")

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_upvector")

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return int(self._mjx_model.nu/4)




  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])


    
def load_ff():
    ff_data = np.load(consts.ROOT_PATH / "FF.npy")
    return jp.array(ff_data)
def load_th():
    th_data = np.load(consts.ROOT_PATH / "TH.npy")
    return jp.array(th_data)

FF_WS = load_ff()
TH_WS = load_th()

class Finger(Enum):
    FF = 1
    MF = 2
    RF = 3
    TH = 4

class FingerTipGoalManager:
    def __init__(self):
        # Point cloud representing FF workspace
        self._FF_WS = FF_WS
        # Point cloud representing TH workspace
        self._TH_WS = TH_WS



    def sample(self, finger:Finger, rng):
        

        if finger == Finger.TH:
            ws = self._TH_WS
        else:
            ws = self._FF_WS
        # jax.debug.print("Sampling from workspace of shape {}", ws.shape)

        
        indices = jax.random.choice(rng, 10, shape=(1,), replace=False)
        # jax.debug.print("Sampling from workspace with index {}", indices)

        goal_xyz = ws[indices]
      
        # Apply offsets for non-TH fingers
        if finger == Finger.MF:
            goal_xyz = goal_xyz + jp.array([-0.00009, -0.0908, 0.0])
        elif finger == Finger.RF:
            goal_xyz = goal_xyz + jp.array([-0.0001, -0.0454, 0.0])

        return goal_xyz

    def sample_goal_with_distance(self, finger: Finger, finger_tip_pos, rng, dist_threshold=0.05):
        "sample a goal close to fingertip"
        goal_xyz = self.sample(finger,rng)
        # if finger == Finger.TH:
        #     ws = self._TH_WS
        # else:
        #     ws = self._FF_WS
    
        # # Apply offsets for non-TH fingers
        # if finger == Finger.MF:
        #     ws = ws + jp.array([-0.00009, -0.0908, 0.0])
        # elif finger == Finger.RF:
        #     ws = ws + jp.array([-0.0001, -0.0454, 0.0])
    
        # # Compute distances to fingertip
        # dists = jp.linalg.norm(ws - finger_tip_pos, axis=1)
    
        # # Create mask for close points
        # close_mask = dists < dist_threshold
    
        # # Pad indices to fixed length (same as ws.shape[0])
        # valid_indices = jp.nonzero(close_mask, size=ws.shape[0], fill_value=0)[0]
    
        # # We use ws.shape[0] as a safe upper bound
        # index_to_sample = jax.random.choice(rng, 10, shape=(1,), replace=False)
    
        # # Sample either from full set or from close indices
        # def sample_from_all():
        #     return jax.random.choice(rng, ws.shape[0], shape=(1,), replace=False)
    
        # def sample_from_close():
        #     return valid_indices[index_to_sample]
    
        # # Sum mask to count how many valid points we actually have
        # num_valid = jp.sum(close_mask)
    
        # indices = jax.lax.cond(num_valid > 0, sample_from_close, sample_from_all)
    
        # goal_xyz = ws[indices]
        return goal_xyz


