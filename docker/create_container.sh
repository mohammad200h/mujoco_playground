sudo xhost +local:root

sudo docker run --runtime=nvidia -it --name mujocoplayground \
  -v $(pwd)/../../mujoco_playground:/workspace/mujoco_playground \
  -e DISPLAY -e LOCAL_USER_ID=$(id -u) -e LOCAL_GID=$(id -g) \
  -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e MUJOCO_GL=egl \
  --net=host  --privileged mujocoplayground
