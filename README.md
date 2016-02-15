# cmu-ammml-project
Project for the Advanced Multimodal Machine Learning course at CMU.

## Setup
Bootstrapping is currently only supported for Ubuntu 14.04. You can use either Theano or TensorFlow as the backend. It's not possible to have both in a single setup since they require different CUDA/cuDNN versions. To bootstrap, simply specify the backend and run the bootstrap script.
```bash
$ BACKEND=theano bootstrap/bootstrap_ubuntu.sh
or
$ BACKEND=tensorflow bootstrap/bootstrap_ubuntu.sh
```
If running without GPUs, set the `CPU_ONLY` flag:
```bash
$ BACKEND=theano CPU_ONLY=1 bootstrap/bootstrap_ubuntu.sh
```
Now you can install the python packages to a virtual environment using `install.sh`:
```bash
$ ./install.sh
```
By default, the virtualenv is created inside `./env`. You can change the location using the `VENV_DIR` variable:
```bash
$ VENV_DIR=venv ./install.sh
```
Reboot the system now, and you're ready to go. You can access the virtualenv by sourcing its activate script:
```bash
$ source <VENV_DIR>/bin/activate
```
