FROM bps-prereq:latest

COPY pkgs /tmp/python-packages

ENV NUMBA_CACHE_DIR "/tmp/numba"

USER root
RUN mkdir /tmp/numba && \
    chmod 777 /tmp/numba && \
    bash -c "conda index --verbose /tmp/python-packages && \
    conda install -y --strict-channel-priority bps-l2b_fh_processor && \
    conda clean -afy" && \
    rm -rf /tmp/numba/* && \
    rm -fr /tmp/python-packages && \
    ln -s /opt/conda_folder/envs/env/bin/bps_l2b_fh_processor /usr/local/bin/

USER nobody
CMD /bin/bash
