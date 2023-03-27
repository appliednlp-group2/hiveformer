name = takashiro

build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile . -t hiveformer
	# singularity build hiveformer.sif docker-daemon://hiveformer:latest

run:
	docker run -d -it --gpus all --rm --name ${name}_hiveformer \
			-v `pwd`:/root/hiveformer \
			-v /share/shota.takashiro/hiveformer_dataset:/root/hiveformer_dataset \
			--shm-size=50gb \
			hiveformer:latest

exec:
	docker exec -it $(name)_hiveformer bash