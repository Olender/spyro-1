FROM firedrakeproject/firedrake:2024-09 AS spyro_base

USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

USER firedrake
RUN . /home/firedrake/firedrake/bin/activate; pip3 install wheel --upgrade
RUN . /home/firedrake/firedrake/bin/activate; pip3 install scipy
RUN . /home/firedrake/firedrake/bin/activate; pip3 install matplotlib
RUN . /home/firedrake/firedrake/bin/activate; pip3 install segyio

# spyro dependencies
USER root
RUN apt-get update && apt-get install -y libgmp3-dev libmpfr-dev libcgal-dev python3-tk
USER firedrake
# The PyPI SeismicMesh has bugs. It is better to install from GitHub
RUN . /home/firedrake/firedrake/bin/activate; pip3 install --no-dependencies git+https://github.com/NDF-Poli-USP/SeismicMesh.git
RUN . /home/firedrake/firedrake/bin/activate; pip3 install pyamg
RUN . /home/firedrake/firedrake/bin/activate; pip3 install meshio

FROM spyro_base AS spyro_release

RUN . /home/firedrake/firedrake/bin/activate; pip install git+https://github.com/Olender/spyro-1.git

FROM spyro_base AS spyro_development

# For notebook development
RUN . /home/firedrake/firedrake/bin/activate; pip3 install notebook
EXPOSE 8888

RUN . /home/firedrake/firedrake/bin/activate; pip3 install pytest
WORKDIR /home/firedrake
RUN echo "/home/firedrake/shared/spyro" >> /home/firedrake/firedrake/lib/python3.12/site-packages/shared.pth
