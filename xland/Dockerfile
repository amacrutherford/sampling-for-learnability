FROM nvcr.io/nvidia/jax:24.04-py3

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

USER root

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

RUN pip install git+https://github.com/DramaCow/jaxued.git
RUN pip install git+https://github.com/corl-team/xland-minigrid.git@9ddd6116a8a5ece792ccd41a29d43b5b12df16b5
RUN pip install safetensors

USER ${MYUSER}

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter 
# RUN pip install jupyterlab

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/${MYUSER}
