#!/usr/bin/env bash

set -e

DEFAULT_VENV_DIR="env"
TENSORFLOW_PKG_DIR="${HOME}/tensorflow/pkg"

if ! [ ${VENV_DIR+x} ]; then
    VENV_DIR=${DEFAULT_VENV_DIR}
fi

python2 -m virtualenv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip
pip install --upgrade six
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade keras
pip install --upgrade cython
pip install --upgrade h5py
pip install --upgrade nltk
if [ -d "${TENSORFLOW_PKG_DIR}" ]; then
    pip install --upgrade \
        ${TENSORFLOW_PKG_DIR}/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
fi
pip install --upgrade ipython
