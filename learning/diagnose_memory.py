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
"""Diagnose memory usage of mjx.Data for batched environments."""

from absl import app
from absl import flags
import jax
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground._src import memory_utils

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "Name of the environment to diagnose.",
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs",
    8192,
    "Number of batched environments to estimate memory for.",
)


def main(argv):
  del argv  # unused

  # Load environment
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides={"impl": "jax"}
  )

  # Reset to get initial state
  rng = jax.random.PRNGKey(0)
  state = env.reset(rng)

  # Print memory report
  memory_utils.print_memory_report(state.data, num_envs=_NUM_ENVS.value)

  # Get optimization suggestions
  suggestions = memory_utils.suggest_optimizations(
      state.data, _NUM_ENVS.value
  )
  if suggestions:
    print("\nOPTIMIZATION SUGGESTIONS:")
    for i, suggestion in enumerate(suggestions, 1):
      print(f"  {i}. {suggestion}")

  # Additional tips
  print("\n" + "=" * 80)
  print("ADDITIONAL MEMORY OPTIMIZATION TIPS:")
  print("=" * 80)
  print("""
  1. Reduce nconmax/njmax in make_data():
     - These parameters control max contacts/joints
     - If your model doesn't need full capacity, reduce them
     - Example: mjx_env.make_data(model, nconmax=50, njmax=100)

  2. Use smaller batch sizes:
     - Consider using fewer environments (e.g., 4096 instead of 8192)
     - Or use gradient accumulation to maintain effective batch size

  3. Check JAX memory settings:
     - Set XLA_PYTHON_CLIENT_MEM_FRACTION to limit GPU memory
     - Example: os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

  4. Profile actual GPU memory:
     - Use nvidia-smi or jax.profiler to see actual usage
     - Compare with estimates to find bottlenecks

  5. Consider using CPU for some operations:
     - Move non-critical data to CPU with jax.device_put(..., jax.devices('cpu')[0])
     - Only keep active simulation data on GPU
  """)


if __name__ == "__main__":
  app.run(main)

