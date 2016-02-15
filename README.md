# cmu-ammml-project
Project for the Advanced Multimodal Machine Learning course at CMU.

## Setup
### Install required libraries
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
Reboot the system after this so drivers are properly loaded.

### Install Python packages
Once the required libraries have been installed (either through the bootstrap script or manually), you can install the Python packages to a virtual environment using `install.sh`:
```bash
$ ./install.sh
```
By default, the virtualenv is created inside `./env`. You can change the location using the `VENV_DIR` variable:
```bash
$ VENV_DIR=venv ./install.sh
```
You're ready to go now. You can access the virtualenv by sourcing its activate script:
```bash
$ source <VENV_DIR>/bin/activate
```
