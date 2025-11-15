# Base image
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Accept build arguments for user configuration
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=user

# Install all required packages in a single layer to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libx11-dev \
        x11-apps \
        python3-tk \
        build-essential \
        cmake \
        libhdf5-dev \
        libvtk7-dev \
        libboost-all-dev \
        libcgal-dev \
        libtinyxml-dev \
        qtbase5-dev \
        libvtk7-qt-dev \
        python3-dev \
        pkg-config \
        octave \
        liboctave-dev \
        gengetopt \
        help2man \
        groff \
        pod2pdf \
        bison \
        flex \
        libhpdf-dev \
        libtool \
        python3.11 \
        python3-pip \
        emacs-nox && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir setuptools==58.2.0 && \
    pip install --no-cache-dir numpy matplotlib Cython pyproject-toml h5py

# Set the working directory
WORKDIR /app

# Create a user with matching UID/GID before installing openEMS
# This ensures proper ownership of all files
RUN if getent group ${GROUP_ID} > /dev/null 2>&1; then \
        existing_group=$(getent group ${GROUP_ID} | cut -d: -f1); \
        if [ "$existing_group" != "${USER_NAME}" ]; then \
            groupmod -n ${USER_NAME} $existing_group; \
        fi; \
    else \
        groupadd -g ${GROUP_ID} ${USER_NAME}; \
    fi && \
    if getent passwd ${USER_ID} > /dev/null 2>&1; then \
        existing_user=$(getent passwd ${USER_ID} | cut -d: -f1); \
        if [ "$existing_user" != "${USER_NAME}" ]; then \
            usermod -l ${USER_NAME} -g ${GROUP_ID} -m -d /home/${USER_NAME} $existing_user || \
            useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}; \
        fi; \
    else \
        useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}; \
    fi

# Install openEMS as root, then change ownership
# Use explicit path to user's home directory instead of ~
RUN git clone --recursive https://github.com/thliebig/openEMS-Project.git && \
    cd openEMS-Project && \
    ./update_openEMS.sh /home/${USER_NAME}/opt/openEMS --with-hyp2mat --with-CTB --python && \
    cd /app && \
    chown -R ${USER_ID}:${GROUP_ID} /app /home/${USER_NAME}

# Copy your Python script into the container
COPY --chown=${USER_ID}:${GROUP_ID} src/Simple_Patch_Antenna.py /app/

# Switch to the created user
USER ${USER_NAME}

# Default command
CMD ["bash"]