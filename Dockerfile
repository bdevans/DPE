# docker build -t pe .
# docker run -it -v pe:/usr/pe/data --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name pe pe

FROM continuumio/miniconda3

LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT [ "/bin/bash", "-c" ]

#EXPOSE 5000

# Use the environment.yml to create the conda environment.
# https://fmgdata.kinja.com/using-docker-with-conda-environments-1790901398
RUN [ "conda", "update", "conda", "-y" ]
RUN [ "conda", "update", "--all", "-y" ]
COPY environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN [ "conda", "env", "create" ]

# Use bash to source our new environment for setting up
# private dependencies—note that /bin/bash is called in
# exec mode directly
WORKDIR /usr/pe
#RUN [ "/bin/bash", "-c", "source activate pe && python setup.py develop" ]

# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY data /usr/pe/data/
COPY *.py /usr/pe/
VOLUME /usr/pe/results

# ENV PYTHONWARNINGS=ignore
# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
CMD [ "source activate pe && exec python characterise.py" ]
# -Wignore::DeprecationWarning
