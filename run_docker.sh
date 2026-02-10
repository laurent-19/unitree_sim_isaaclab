# Allow X server access from Docker
xhost +local:docker

sudo docker run --gpus all -it --rm \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e QT_X11_NO_MITSHM=1 \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -v /usr/share/vulkan:/usr/share/vulkan:ro \
  -v /usr/share/glvnd:/usr/share/glvnd:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /home/analog/develop/unitree_sim_isaaclab:/home/code/unitree_sim_isaaclab \
  -v /home/analog/develop/inspire_hand_ws:/home/code/inspire_hand_ws \
  unitree-sim:latest /bin/bash
