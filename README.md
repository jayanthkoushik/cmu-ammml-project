# cmu-ammml-project
Project for the Advanced Multimodal Machine Learning course at CMU.

## Setup
Bootstraping is currently only supported for Ubuntu 14.04. Simply run the bootstrap script:
```bash
$ bootstrap/bootstrap_ubuntu.sh
```
If running without GPUs, set the `CPU_ONLY` flag:
```bash
$ CPU_ONLY=1 bootstrap/bootstrap_ubuntu.sh
```
Source `bashrc` to update the environment variables:
```bash
$ source ~/.bashrc
```
Now you can install the python packages to a virtual environment using `install.sh`:
```bash
$ ./install.sh
or
$ VENV_DIR=venv ./install.sh
```
Reboot the system now, and you're ready to go.
