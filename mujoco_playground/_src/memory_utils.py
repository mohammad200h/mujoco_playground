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
"""Utilities for measuring and optimizing memory usage of mjx.Data."""

import jax
from mujoco import mjx


def get_data_size(data: mjx.Data) -> int:
  """Get total memory size in bytes of mjx.Data object.
  
  Args:
    data: mjx.Data object to measure
    
  Returns:
    Total memory size in bytes
  """
  def array_size(x):
    if isinstance(x, jax.Array):
      return x.size * x.dtype.itemsize
    return 0
  
  sizes = jax.tree_util.tree_map(array_size, data)
  total_size = sum(jax.tree_util.tree_leaves(sizes))
  return total_size


def get_data_size_breakdown(data: mjx.Data) -> dict:
  """Get detailed memory size breakdown of mjx.Data object.
  
  Args:
    data: mjx.Data object to analyze
    
  Returns:
    Dictionary with total size and breakdown by field
  """
  def array_info(path, x):
    if isinstance(x, jax.Array):
      size_bytes = x.size * x.dtype.itemsize
      return {
          'path': '/'.join(str(p) for p in path),
          'shape': x.shape,
          'dtype': str(x.dtype),
          'size_bytes': size_bytes
      }
    return None
  
  leaves, treedef = jax.tree_util.tree_flatten_with_path(data)
  info_list = [array_info(path, leaf) for path, leaf in leaves]
  info_list = [info for info in info_list if info is not None]
  
  # Sort by size (largest first)
  info_list.sort(key=lambda x: x['size_bytes'], reverse=True)
  
  total_size = sum(info['size_bytes'] for info in info_list)
  
  return {
      'total_bytes': total_size,
      'total_mb': total_size / (1024**2),
      'total_gb': total_size / (1024**3),
      'num_arrays': len(info_list),
      'breakdown': info_list
  }


def estimate_batched_memory(
    single_data: mjx.Data,
    num_envs: int,
    include_state: bool = True
) -> dict:
  """Estimate memory usage for batched environments.
  
  Args:
    single_data: mjx.Data from a single environment
    num_envs: Number of batched environments
    include_state: Whether to include State wrapper memory (obs, reward, etc.)
    
  Returns:
    Dictionary with memory estimates
  """
  single_size = get_data_size(single_data)
  batched_data_size = single_size * num_envs
  
  # Estimate State wrapper overhead (obs, reward, done, metrics, info)
  # This is approximate - actual size depends on observation space
  state_overhead_per_env = 0
  if include_state:
    # Rough estimate: obs is typically similar size to qpos+qvel
    # Add some overhead for reward, done, metrics, info
    state_overhead_per_env = single_size * 0.1  # 10% overhead estimate
  
  total_estimated = batched_data_size + (state_overhead_per_env * num_envs)
  
  return {
      'single_env_data_mb': single_size / (1024**2),
      'batched_data_mb': batched_data_size / (1024**2),
      'batched_data_gb': batched_data_size / (1024**3),
      'total_estimated_mb': total_estimated / (1024**2),
      'total_estimated_gb': total_estimated / (1024**3),
      'num_envs': num_envs,
      'top_fields': get_data_size_breakdown(single_data)['breakdown'][:10]
  }


def print_memory_report(data: mjx.Data, num_envs: int = 1):
  """Print a detailed memory usage report.
  
  Args:
    data: mjx.Data object to analyze
    num_envs: Number of environments (for batched estimate)
  """
  breakdown = get_data_size_breakdown(data)
  
  print("=" * 80)
  print("MJX DATA MEMORY USAGE REPORT")
  print("=" * 80)
  print(f"\nSingle Environment:")
  print(f"  Total: {breakdown['total_mb']:.2f} MB ({breakdown['total_gb']:.4f} GB)")
  print(f"  Number of arrays: {breakdown['num_arrays']}")
  
  if num_envs > 1:
    estimate = estimate_batched_memory(data, num_envs)
    print(f"\nBatched ({num_envs} environments):")
    print(f"  Data only: {estimate['batched_data_gb']:.2f} GB")
    print(f"  Total estimated: {estimate['total_estimated_gb']:.2f} GB")
    print(f"\n  ⚠️  WARNING: With {num_envs} envs, you need ~{estimate['total_estimated_gb']:.2f} GB GPU memory just for mjx.Data!")
  
  print(f"\nTop 10 Largest Arrays:")
  print(f"{'Field':<40} {'Shape':<25} {'Dtype':<10} {'Size (MB)':<12}")
  print("-" * 80)
  for item in breakdown['breakdown'][:10]:
    shape_str = str(item['shape'])
    print(f"{item['path']:<40} {shape_str:<25} {item['dtype']:<10} {item['size_bytes']/(1024**2):>10.4f}")
  
  print("\n" + "=" * 80)


def suggest_optimizations(data: mjx.Data, num_envs: int) -> list:
  """Suggest memory optimization strategies.
  
  Args:
    data: mjx.Data object to analyze
    num_envs: Number of batched environments
    
  Returns:
    List of optimization suggestions
  """
  suggestions = []
  breakdown = get_data_size_breakdown(data)
  estimate = estimate_batched_memory(data, num_envs)
  
  # Check if memory usage is high
  if estimate['batched_data_gb'] > 8:
    suggestions.append(
        f"⚠️  High memory usage ({estimate['batched_data_gb']:.2f} GB). "
        "Consider reducing num_envs or optimizing model."
    )
  
  # Check for large arrays
  large_arrays = [item for item in breakdown['breakdown'] 
                  if item['size_bytes'] > 10 * 1024 * 1024]  # > 10MB
  if large_arrays:
    suggestions.append(
        f"Found {len(large_arrays)} large arrays (>10MB each). "
        "Check if all are necessary for your use case."
    )
  
  # Check nconmax/njmax
  # These are set in make_data and can significantly impact memory
  suggestions.append(
      "💡 Consider reducing nconmax/njmax in make_data() if your model "
      "doesn't need the full contact/joint capacity."
  )
  
  # Suggest reducing batch size
  if estimate['batched_data_gb'] > 16:
    suggested_envs = int(num_envs * 8 / estimate['batched_data_gb'])
    suggestions.append(
        f"💡 For 8GB GPU, consider reducing num_envs from {num_envs} to ~{suggested_envs}"
    )
  
  # Suggest using CPU for some operations
  suggestions.append(
      "💡 Consider using jax.device_put to move some data to CPU if not needed on GPU"
  )
  
  return suggestions

