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
from mujoco_playground._src import collision
from mujoco_playground._src.manipulation.leap_grasp import leap_grasp_constants as consts

from typing import Literal


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

    # cube following trajectory
    self.x_force_applied = False
    self.desired_y_force = 0.1
    self.force = 0.05


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

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])

  # Contact
  def there_is_contact_between_th_and_object(self, data: mjx.Data)-> jax.Array:

    th_geoms = ["th_mp_collision","th_bs_collision_1","th_px_collision_1",
                "th_ds_collision_1","th_tip"]
    return jp.array([
      collision.geoms_colliding(data, self._mj_model.geom(g_name).id,
                                self._mj_model.geom("cube").id)
      for g_name in th_geoms
    ])

  def there_is_contact_for_ff_mf_rf(self,finger:Literal["if", "mf", "rf"], data:mjx.Data)-> jax.Array:
    prefixes = ["_bs_collision_1","_px_collision","_md_collision_1",
                "_md_collision_5","_ds_collision_1","_tip"]
    finger_geoms = [finger + prefix for prefix in prefixes]
    return jp.array([
      collision.geoms_colliding(data, self._mj_model.geom(g_name).id,
                                self._mj_model.geom("cube").id)
      for g_name in finger_geoms
    ])

  # Cube following trajectory
  def get_cube_direction(self,data: mjx.Data):
    cube_vel = mjx_env.get_sensor_data(self.mj_model , data , "cube_linvel" )
    v = cube_vel / jp.linalg.norm(cube_vel) if jp.linalg.norm(cube_vel) > 0 else cube_vel

    return v

  def get_vector_pointing_to_palm(self, data: mjx.Data):
    cube_pos = mjx_env.get_sensor_data(self.mj_model , data , "cube_position" )
    palm_pos = mjx_env.get_sensor_data(self.mj_model , data , "palm_position" )
    v = palm_pos - cube_pos
    v = v / jp.linalg.norm(v) if jp.linalg.norm(v) > 0 else v

    return v

  def get_impulse(self):
    # com = data.xipos[self._mj_model.geom("cube").id]
    impulse = jp.array([self.desired_y_force,0,0,0,0,0])
    xfrc_applied = jp.zeros(16)

    return  jp.concatenate([xfrc_applied, impulse])

  def head_toward_the_hand(self, data: mjx.Data):
    t = data.time
    if t > 2*self.dt:
      data.qfrc_applied[:] = 0






  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

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
