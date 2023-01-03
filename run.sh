#!/bin/bash

export containerName=yolokeras_$USER
sleep 3 && \
        xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerName` >/dev/null 2>&1 &
docker run -d --gpus '"device=0"' --rm -it \
	--volume="/home/mbenavent/workspace:/workspace:rw" \
	--volume="/mnt/md1/datasets/:/datasets:ro" \
	--volume=$HOME/.Xauthority:/root/.Xauthority:ro \
	--workdir="/workspace" \
	-v $XSOCK:$XSOCK:rw \
	--net=host \
        --env="DISPLAY" \
	--name $containerName \
	mbenavent/yolokeras:latest bash
