FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gmsh \
        git \
        gnupg \
        python3 \
        python3-gmsh \
        python3-matplotlib \
        python3-numpy \
    && curl -fsSL https://dl.openfoam.com/add-debian-repo.sh | bash \
    && apt-get update \
    && apt-get install -y --no-install-recommends openfoam2512-default \
    && rm -rf /var/lib/apt/lists/*

ENV WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2512

RUN printf '%s\n' \
    'source /usr/lib/openfoam/openfoam2512/etc/bashrc' \
    > /etc/profile.d/openfoam.sh
