FROM    nvidia/cuda:11.2-devel-ubuntu20.04
LABEL   maintainer="Louis Ross <louis.ross@gmail.com"

ARG     MYDIR=/home/spiking-model-gpu
WORKDIR ${MYDIR}

COPY    install-deps ${MYDIR}/
COPY    bmtk.zip ${MYDIR}/

RUN     echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN     bash ${MYDIR}/install-deps ${MYDIR} >>install-deps.log
CMD     ["bash"]
