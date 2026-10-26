# Plano B: PX4 SITL + Gazebo Harmonic (gz-sim 8) para arm64 (DGX Spark), headless (EGL), sem GUI.
# NAO construido nesta etapa (CPU do Spark ocupada). Construir com:
#   cd ~/cerise2026/drones/sim/gazebo && sudo nice -n 19 docker build --build-arg JOBS=4 -t cerise26_gzpx4:latest .
# Estimativa (nao medida): apt ~1,5-2 GB + clone do PX4 ~1 GB; compilacao px4_sitl_default ~8-12 min com -j16
# e CPU livre, ~30-45 min com -j4 e o Spark carregado; imagem final ~6-8 GB.
FROM ubuntu:22.04
ARG PX4_TAG=v1.15.4
ARG JOBS=4
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Gazebo Harmonic (repositorio oficial da OSRF) + bindings Python do gz-transport/gz-msgs
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release git sudo wget python3-pip python3-dev python3-venv \
      build-essential cmake ninja-build ccache astyle file zip unzip rsync libxml2-utils \
      libeigen3-dev libopencv-dev protobuf-compiler libgstreamer-plugins-base1.0-dev \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
         > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends gz-harmonic python3-gz-transport13 python3-gz-msgs10 \
    && rm -rf /var/lib/apt/lists/*

# PX4-Autopilot (GitHub oficial), versao fixa
RUN git clone --depth 1 --branch ${PX4_TAG} --recursive --shallow-submodules https://github.com/PX4/PX4-Autopilot.git /px4
RUN pip3 install --no-cache-dir -r /px4/Tools/setup/requirements.txt \
    && pip3 install --no-cache-dir "mavsdk>=2,<3" numpy pillow
# SITL (inclui o gz_bridge do PX4 para o Gazebo Harmonic); nice + poucos jobs por causa da producao
RUN cmake -S /px4 -B /px4/build/px4_sitl_default -G Ninja -DCONFIG=px4_sitl_default \
    && nice -n 19 ninja -C /px4/build/px4_sitl_default -j ${JOBS} \
    && (ccache -C || true)

ENV PX4_DIR=/px4 \
    GZ_SIM_RESOURCE_PATH=/mundo/modelos:/px4/Tools/simulation/gz/models:/px4/Tools/simulation/gz/worlds \
    GZ_SIM_SYSTEM_PLUGIN_PATH=/px4/build/px4_sitl_default/src/modules/simulation/gz_plugins \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1
WORKDIR /px4
