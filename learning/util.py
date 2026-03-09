import jax
import jax.numpy as jnp
import jax.tree_util

def inspect_state_data(state):

  # Inspect all properties of state.data using JAX tree utilities
  print("\n=== Available properties of state.data ===")
  def inspect_field(path, value):
    if hasattr(value, 'shape'):
      path_str = '/'.join(str(p) for p in path) if path else 'root'
      print(f"  {path_str}: shape={value.shape}, dtype={value.dtype}")
    return value
  
  # Use tree_map_with_path to inspect all fields
  jax.tree_util.tree_map_with_path(inspect_field, state.data)
  

def check_state_for_nan(state):
  if jnp.any(jnp.isnan(state.data.qpos)) or jnp.any(jnp.isnan(state.data.qvel)):
    print("ERROR: NaN detected in state.data.qpos or state.data.qvel!")
    raise ValueError("NaN detected in state.data.qpos or state.data.qvel")
  if jnp.any(jnp.isnan(state.reward)) or jnp.any(jnp.isnan(state.done)):
    print("ERROR: NaN detected in state.reward or state.done!")
    raise ValueError("NaN detected in state.reward or state.done")
  if jnp.any(jnp.isnan(state.obs["state"])):
    print("ERROR: NaN detected in state.obs!")
    raise ValueError("NaN detected in state.obs")
  if jnp.any(jnp.isnan(state.obs["privileged_state"])):
    print("ERROR: NaN detected in state.obs['privileged_state']!")
    raise ValueError("NaN detected in state.obs['privileged_state']")