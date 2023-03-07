#! /bin/bash 

# install docker 
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
sudo apt-get update
sudo systemctl restart docker


