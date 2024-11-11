docker run 	--rm \
		--network host \
		--gpus all \
		-it \
		--privileged \
		-v /tmp/.x11-unix:/tmp/.x11-unix \
		-v /home/camille/IL/diffusion_policy:/app \
		-v /media/camille/SATA:/relative_folder/SATA \
		-v /home/camille/sim:/relative_folder/sim \
		-w /app \
		-e DISPLAY=$DISPLAY \
		-e http_proxy="http://172.17.0.1:7890" \
		-e https_proxy="http://172.17.0.1:7890" \
		-e all_proxy="http://172.17.0.1:7890" \
		nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04-diffusion_ros \
		/usr/bin/zsh
