#!/bin/bash
docker run -it \
--rm \
--gpus all \
--runtime=nvidia \
--name tensorrt \
--shm-size=2gb \
-v $(pwd)/:/workspace/EfficientBioAI/ \
efficient_bio_ai:v1 \
/bin/bash