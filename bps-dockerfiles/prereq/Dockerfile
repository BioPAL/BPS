FROM registry.access.redhat.com/ubi8-minimal:latest

COPY pkgs /tmp/python-packages

ENV LANG=en_US.utf8
ENV BASH_ENV="/etc/profile"
ENV PROJ_DATA="/opt/conda_folder/envs/env/share/proj"

USER root
RUN cd /tmp && \
    curl -fsSL https://github.com/conda-forge/miniforge/releases/download/25.1.1-2/Miniforge3-Linux-x86_64.sh -o Miniforge3.sh && \
    chmod +x Miniforge3.sh && \
    ./Miniforge3.sh -b -p /opt/conda_folder && \
    /opt/conda_folder/bin/conda install -y -c conda-forge conda-index && \
    /opt/conda_folder/bin/conda index --verbose /tmp/python-packages && \
    /opt/conda_folder/bin/conda config --prepend channels conda-forge && \
    /opt/conda_folder/bin/conda config --prepend channels /tmp/python-packages && \
    /opt/conda_folder/bin/conda create -n env -y python=3.12 bps-common bps-transcoder && \
    /opt/conda_folder/bin/conda clean -afy && \
    cp /opt/conda_folder/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "conda activate env" >> /etc/profile.d/conda.sh && \
    rm -rf /tmp/* && \
    ln -s /opt/conda_folder/envs/env/bin/getfattr /usr/local/bin/
