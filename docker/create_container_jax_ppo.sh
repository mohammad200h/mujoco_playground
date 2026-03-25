sudo xhost +local:root

# Accept GPU_ID as first argument (e.g. ./create_container.sh 1), default 0
# Must be a number so CUDA_VISIBLE_DEVICES is valid.
RAW_ID=${1:-0}
if ! [[ "$RAW_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [GPU_ID]" >&2
  echo "  GPU_ID must be a number (e.g. 0 or 1). You passed: $RAW_ID" >&2
  echo "  Example: $0 1" >&2
  exit 1
fi
GPU_ID=$RAW_ID


MULTI_GPU=${MULTI_GPU:-false}

if  [[ "$MULTI_GPU" == "true" ]]; then
  echo "Running in multi-GPU mode"
  CONTAINER_NAME=mujocoplayground_jax_ppo_multi_gpu

  sudo docker run --runtime=nvidia -it --name ${CONTAINER_NAME} \
      -e DISPLAY -e LOCAL_USER_ID=$(id -u) -e LOCAL_GID=$(id -g) \
      -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -e MUJOCO_GL=egl \
      --net=host  --privileged mujocoplayground_jax_ppo
else
  echo "Running in single-GPU mode"
  CONTAINER_NAME=mujocoplayground_jax_ppo_${GPU_ID}

  sudo docker run --runtime=nvidia -it --name ${CONTAINER_NAME} \
    -e DISPLAY -e LOCAL_USER_ID=$(id -u) -e LOCAL_GID=$(id -g) \
    -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e MUJOCO_GL=egl \
    -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
    -e MUJOCO_PLAYGROUND_WARP_SINGLE_GPU=${GPU_ID} \
    --net=host  --privileged mujocoplayground_jax_ppo

fi