FROM bps-prereq:latest

COPY lib/* /usr/local/lib/
COPY pkgs /tmp/python-packages

ENV LD_LIBRARY_PATH /usr/local/lib

USER root
RUN microdnf install libgomp && \
    microdnf clean all && \
    bash -c "conda index --verbose /tmp/python-packages && \
    conda install -y --strict-channel-priority bps-l1_framing_processor && \
    conda clean -afy" && \
    rm -fr /tmp/python-packages && \
    ln -s /opt/conda_folder/envs/env/bin/bps_l1_framing_processor /usr/local/bin/

USER nobody
CMD /bin/bash
