# docker build -t dpe .
# docker run -it -v dpe:/usr/dpe/data --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name dpe dpe

FROM continuumio/miniconda3

LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT [ "/bin/bash", "-c" ]

#EXPOSE 5000

# Conda supports delegating to pip to install dependencies
# that aren’t available in anaconda or need to be compiled
# for other reasons. In our case, we need psycopg compiled
# with SSL support. These commands install prereqs necessary
# to build psycopg.
#RUN apt-get update && apt-get install -y \
# libpq-dev \
# build-essential \
#&& rm -rf /var/lib/apt/lists/*

# Use the environment.yml to create the conda environment.
RUN [ "conda", "update", "conda", "-y" ]
RUN [ "conda", "update", "--all", "-y" ]
COPY environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN [ "conda", "env", "create" ]

# Use bash to source our new environment for setting up
# private dependencies—note that /bin/bash is called in
# exec mode directly
WORKDIR /usr/dpe
#RUN [ "/bin/bash", "-c", "source activate dpe && python setup.py develop" ]

# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY bootstrap.py /usr/dpe/
COPY data/biobank_mix_WTCC_ref.csv /usr/dpe/data/
VOLUME /usr/dpe/results

# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
CMD [ "source activate dpe && exec python bootstrap.py" ]
