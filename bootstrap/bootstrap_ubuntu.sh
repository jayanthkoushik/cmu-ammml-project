#!/usr/bin/env bash

set -e

INSTALLERS_DIR="./external"
CUDA_INSTALLER="${INSTALLERS_DIR}/cuda-repo-ubuntu1404_7.0-28_amd64.deb"
CUDA_HOME="/usr/local/cuda"
CUDNN_VERSION="cudnn-6.5-linux-x64-v2"
CUDNN_INSTALLER="${INSTALLERS_DIR}/${CUDNN_VERSION}.tgz"
BAZEL_VERSION="bazel-0.1.4-installer-linux-x86_64.sh"
BAZEL_INSTALLER_URL="https://github.com/bazelbuild/bazel/releases/download/"`
    `"0.1.4/${BAZEL_VERSION}"
PROFILE_FILE="${HOME}/.bashrc"
TENSORFLOW_PKG_DIR="${HOME}/tensorflow/pkg"

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install gcc g++ gfortran build-essential libblas-dev \
    liblapack-dev libatlas-base-dev libhdf5-dev git python-dev python-pip \
    software-properties-common pkg-config zip zlib1g-dev unzip swig \
    libfreetype6-dev libxft-dev libncurses-dev linux-headers-generic \
    linux-image-extra-virtual python-numpy wget

echo | sudo add-apt-repository ppa:webupd8team/java
sudo apt-get -y update
echo debconf shared/accepted-oracle-license-v1-1 select true | \
    sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | \
    sudo debconf-set-selections
sudo apt-get -y install oracle-java8-installer

wget ${BAZEL_INSTALLER_URL}
chmod +x ${BAZEL_VERSION}
${BAZEL_VERSION} --user
echo "export PATH=\$PATH:$HOME/bin" >> ${PROFILE_FILE}
rm ${BAZEL_VERSION}

if ! [ ${CPU_ONLY+x} ]; then
    sudo dpkg -i ${CUDA_INSTALLER}
    sudo apt-get -y update
    sudo apt-get -y install cuda-7.0
    echo "export CUDA_HOME=${CUDA_HOME}" >> ${PROFILE_FILE}
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${CUDA_HOME}/lib64" \
        >> ${PROFILE_FILE}
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"

    tar xvzf ${CUDNN_INSTALLER} -C ~/
    sudo cp ~/${CUDNN_VERSION}/cudnn.h ${CUDA_HOME}/include
    sudo cp ~/${CUDNN_VERSION}/libcudnn* ${CUDA_HOME}/lib64
    sudo chmod a+r ${CUDA_HOME}/lib64/libcudnn*
    rm -r ~/${CUDNN_VERSION}
fi

sudo python2 -m pip install --upgrade virtualenv

echo "export KERAS_BACKEND=tensorflow" >> ${PROFILE_FILE}

cd ~
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
if [ ${CPU_ONLY+x} ]; then
    TF_UNOFFICIAL_SETTING=1 ./configure <<< $'\n\n'
    ${HOME}/bin/bazel build -c opt \
        //tensorflow/tools/pip_package:build_pip_package
else
    TF_UNOFFICIAL_SETTING=1 ./configure <<< $'\ny\n\n\n\n\n3.0\n'
    ${HOME}/bin/bazel build -c opt --config=cuda \
        tensorflow/tools/pip_package:build_pip_package
fi
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TENSORFLOW_PKG_DIR}
