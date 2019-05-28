#!/bin/bash
# Install a clean new copy of Ubuntu 14.04 with default options
# Download CUDNN (Signup required): 
#   https://developer.nvidia.com/rdp/assets/cudnn-70-linux-x64-v3-prod
# SCP CUDNN and this script onto the server and run ./bootstrap.sh

set -e
set -x
CUDA_VERSION=cuda_7.5.18_linux.run
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/$CUDA_VERSION
CUDNN_VERSION=cudnn-7.5-linux-x64-v5.0-rc.tgz

function check_cudnn() {
    if [[ ! -e $CUDNN_VERSION ]]; then
        echo "Warning: Failed to find file $CUDNN_VERSION"
        echo "Download this file from https://developer.nvidia.com (signup required)"
        echo "Press enter to continue or ctrl+C to abort"
        read
    else
        echo "Installing CUDNN from $CUDNN_VERSION"
    fi
}

function install_cudnn() {
    if [[ -e $CUDNN_VERSION ]]; then
        echo "Installing CUDNN from $CUDNN_VERSION"
        pushd /usr/local
        sudo tar xzf $HOME/$CUDNN_VERSION
        echo "Finished installing CUDNN"
        popd
    fi
}

# This should run first to uninstall noveau, then 
function install_cuda() {
    sudo apt-get update

    sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r` build-essential

    wget -nc $CUDA_URL
    chmod 755 $CUDA_VERSION

    tail -F /var/log/nvidia-installer.log &
    sudo ./$CUDA_VERSION --silent --driver --toolkit --samples --no-opengl-libs
    kill %1

    if (nvidia-smi -q | grep -q GPU); then
        echo -e "\nexport PATH=/usr/local/cuda/bin:\$PATH" >> .bashrc
        echo -e "\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> .bashrc
        echo "CUDA installed successfully"
    else
        sudo update-initramfs -u
        clear
        tput setaf 3
        echo "Rebooting in ten seconds. After reboot, run ./$0 again to complete installation."
        tput setaf 15
        sleep 10
        sudo reboot
        exit 1
    fi
}

function install_essential_packages() {
    sudo apt-get update
    sudo apt-get install -y binutils htop git vim
    sudo apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran
    sudo apt-get install -y python-dev
    sudo apt-get install -y python-h5py
    sudo apt-get install -y python-pip
}

function install_pip_virtualenv() {
    sudo pip install virtualenv
    sudo rm -rf venv
    virtualenv venv
    echo "source ~/venv/bin/activate" >> ~/.bashrc
    source ~/.bashrc
}

function install_python_packages() {
    pip install numpy scipy
    pip install keras
    # Tensorflow with GPU support
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
}

if [[ ! `which nvcc` ]]
then
    echo "Could not find nvcc, installing CUDA"
    install_cuda
fi

install_cudnn

install_essential_packages

install_pip_virtualenv

install_python_packages

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n" >> ~/.theanorc
wget -nc "https://raw.githubusercontent.com/lwneal/robot-bernie/master/gputest.py"
python gputest.py

