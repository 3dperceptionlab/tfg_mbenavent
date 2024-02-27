#!/bin/bash

export containerName=yolokeras_$USER
sleep 3 && \
        xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerName` >/dev/null 2>&1 &
docker run -d --gpus '"device=1"' --rm -it \
	--volume="/home/mbenavent/workspace:/workspace:rw" \
	--volume="/mnt/md1/datasets/:/datasets:ro" \
	--workdir="/workspace" \
	--net=host \
	--env="DISPLAY" \
	--name $containerName \
	mbenavent/holoyolo:latest bash

