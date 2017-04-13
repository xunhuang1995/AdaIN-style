FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  sudo

# Run Torch7 installation scripts (dependencies only)
RUN git clone https://github.com/torch/distro.git --depth 1 /root/torch --recursive && cd /root/torch && \
  bash install-deps

# Run Torch7 installation scripts
RUN cd /root/torch && \
# Run without nvcc to prevent timeouts
  sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh && \
  sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh && \
  ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Set ~/torch as working directory
WORKDIR /root/torch

# Restore Torch7 installation script
RUN sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

# Install CUDA libraries
RUN luarocks install cutorch && \
  luarocks install cunn && \
  luarocks install cudnn && \
  luarocks install unsup

# Install RiseML
RUN apt-get install -y --no-install-recommends libssl-dev wget && \
  luarocks install riseml

WORKDIR /opt/AdaIN-style

RUN git clone https://github.com/kuralex/AdaIN-style --depth 1 /opt/AdaIN-style

RUN bash models/download_models.sh

CMD ["th", "demo.lua"]
