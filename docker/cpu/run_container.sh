#!/bin/bash
docker run -it \
--rm \
-u 0 \
--name openvino \
--shm-size=2gb \
-v $(pwd)/:/EfficientBioAI/ \
efficient_bio_ai:v1 \
/bin/bash
-c "apt update && apt install sudo && deployment_tools/demo/demo_security_barrier_camera.sh -d CPU -sample-options -no_show"