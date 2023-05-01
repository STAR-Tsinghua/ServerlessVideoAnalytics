FROM public.ecr.aws/lambda/python:3.8

# Copy function code
COPY ./ ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    ghostscript \
    lcms2-devel \
    libffi-devel \
    libimagequant-devel \
    libjpeg-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    make \
    openjpeg2-devel \
    rh-python38 \
    rh-python38-python-virtualenv \
    sudo \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    xorg-x11-server-Xvfb \
    zlib-devel \
    gcc-c++ \
    python3-devel \
    g++ \
    build-essential \
    python3.8 \
    && yum clean all
# RUN yum -y install gcc-c++ python3-devel
COPY requirements.txt  .
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]