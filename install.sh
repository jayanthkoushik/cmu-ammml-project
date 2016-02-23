#!/usr/bin/env bash

set -e

DEFAULT_VENV_DIR="env"
TENSORFLOW_DIR="${HOME}/tensorflow"
TENSORFLOW_PKG_DIR="${TENSORFLOW_DIR}/pkg"

if ! [ ${VENV_DIR+x} ]; then
    VENV_DIR=${DEFAULT_VENV_DIR}
fi

if ! [ -d ${VENV_DIR} ]; then
    python2 -m virtualenv ${VENV_DIR}
fi
source ${VENV_DIR}/bin/activate

pip install --upgrade pip
pip install --upgrade six
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade cython
pip install --upgrade h5py
pip install --upgrade nltk
pip install --upgrade pydot-ng
pip install --upgrade scikit-learn
pip install --upgrade pillow
if [ -d "${TENSORFLOW_PKG_DIR}"  ]; then
    pip install --upgrade \
        ${TENSORFLOW_PKG_DIR}/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
    rm -rf ${TENSORFLOW_DIR}
else
    pip install --upgrade git+git://github.com/Theano/Theano.git
fi
pip install --upgrade git+git://github.com/fchollet/keras.git
pip install --upgrade ipython

