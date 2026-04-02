FROM bps-prereq:latest

COPY pkgs /tmp/python-packages

USER root
RUN bash -c "conda index --verbose /tmp/python-packages && \
    conda install -y --strict-channel-priority bps-l1_processor && \
    conda clean -afy" && \
    rm -fr /tmp/python-packages && \
    ln -s /opt/conda_folder/envs/env/bin/bps_l1_processor /usr/local/bin/ && \
    ln -s /opt/conda_folder/envs/env/bin/BiomassL0ImportPreProc /usr/local/bin/ && \
    ln -s /opt/conda_folder/envs/env/bin/BPSL1CoreProcessor /usr/local/bin/ && \
    ln -s /opt/conda_folder/envs/env/bin/IonosphericHeightIRI20Estimator /usr/local/bin/

USER nobody
CMD /bin/bash
