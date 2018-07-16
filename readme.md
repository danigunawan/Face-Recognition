 Assuming you have a new Ubuntu system
1) First, check if the device has GPU installed on it using
   sudo lshw -C display
   The GPU will be listed like this 
     description: 3D controller
       product: GM108M [GeForce 940MX]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:01:00.0
       version: a2
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress bus_master cap_list
       configuration: driver=nvidia latency=0
       resources: irq:135 memory:b3000000-b3ffffff memory:a0000000-afffffff memory:b0000000-b1ffffff ioport:4000(size=128)
    NOTE: usually the gpu_id is 0 but do it to confirm if the GPU is NVidia as doing without it can be costly (would definitely require SageMaker or buying an eGPU, docker, pci express connectors)
 2) Check if the GPU is compatible with CUDA here.
    https://developer.nvidia.com/cuda-gpus 
    Or Google the GPU with family name, go to the product page and see features like this one
    https://developer.nvidia.com/cuda-gpus
 3) Install CUDA if enabled.
    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
    Contains information to install CUDA's latest version. 
    If you wish to install legacy version of CUDA, refer to the footer(why? as TensorFlow requires CUDA 8.0 and doesn't work with anything newer since 2017
    MxNet works fine with Cuda 9.2.# You can skip Power9 setup mentioned on the site 
    Commands for quick access:
    3.1)uname -m && cat /etc/*release 
         to check if there is a compatible version of Ubuntu
    3.2)gcc --version
          to check if C compiler is installed
          sudo apt-get install gcc-4.9 g++-4.9
          sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
          To set to gcc 4.9, if newer version exists,replace 4.9 with the version id of the newer version.
    3.3)The kernel headers and development packages for the currently running kernel can be installed with:
        sudo apt-get install linux-headers-$(uname -r)
    3.4)https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal
        Download the latest CUDA from here.Doesn't require account on NVidia but make one nonetheless to save time for later.
    3.5)Use this command to confirm  correct thing is installed
        md5sum cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
        Get the output and compare with this corresponding information present on this site:
        https://developer.download.nvidia.com/compute/cuda/9.2/Prod/docs/sidebar/md5sum.txt
        (Tip: Copy the output, search the location's checksums on your web browser and if exact same thing is found, then it is all good to go. This is a formality but should be done nonetheless. If something wrong, use this command to immediately uninstall and repeat from step 3.3 onwards
        sudo apt-get --purge remove <package_name>)
    3.6)Install repository meta-data
         sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
    3.7) Installing the CUDA public GPG key
         sudo apt-key add /var/cuda-repo-9-2/7fa2af80.pub
    3.8) sudo apt-get update
    3.9) sudo apt-get install cuda
 4)The PATH variable needs to include /usr/local/cuda-9.2/bin 
   To add this path to the PATH variable:
   export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
   In addition, when using the runfile installation method, the LD_LIBRARY_PATH variable needs to contain /usr/local/cuda-9.2/lib64 on a 64-bit system
   To change the environment variables for 64-bit operating systems:
   $ export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
 5) You can skip Power9 setup mentioned on the site
 6)NVIDIA is providing a user-space daemon on Linux to support persistence of driver state across CUDA job runs. The daemon approach provides a m ore elegant and robust solution to this problem than persistence mode. For more details on the NVIDIA Persistence Daemon, see the documentation here.The NVIDIA Persistence Daemon can be started as the root user by running:
  /usr/bin/nvidia-persistenced --verbose
 7) Install Writable Samples 
    In order to modify, compile, and run the samples, the samples must be installed with write permissions. A convenience installation script is provided:
    cuda-install-samples-9.2.sh <dir>
    #absolute path of your choice
 8)  Verify the Driver Version
     If you installed the driver, verify that the correct version of it is loaded. If you did not install the driver, or are using an operating system where the driver is not loaded via a kernel module, such as L4T, skip this step.
     When the driver is loaded, the driver version can be found by executing the command
    cat /proc/driver/nvidia/version
 9)The version of the CUDA Toolkit can be checked by running nvcc -V in a terminal window. The nvcc command runs the compiler driver that compiles  CUDA programs. It calls the gcc compiler for C code and the NVIDIA PTX compiler for the CUDA code.
 The NVIDIA CUDA Toolkit includes sample programs in source form. You should compile them by changing to ~/NVIDIA_CUDA-9.2_Samples and typing make. The resulting binaries will be placed under ~/NVIDIA_CUDA-9.2_Samples/bin.
 10)After compilation, find and run deviceQuery under ~/NVIDIA_CUDA-9.2_Samples. If the CUDA software is installed and configured correctly, the output for deviceQuery should look similar to that shown in here 
  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/graphics/valid-results-from-sample-cuda-devicequery-program.png
 11) The exact appearance and the output lines might be different on your system. The important outcomes are that a device was found (the first
    highlighted line), that the device matches the one on your system (the second highlighted line), and that the test passed (the final highlighted line).If a CUDA-capable device and the CUDA Driver are installed but deviceQuery reports that no CUDA-capable devices are present, this likely means that the /dev/nvidia* files are missing or have the wrong permissions.
    On systems where SELinux is enabled, you might need to temporarily disable this security feature to run deviceQuery. To do this, type:
    sudo setenforce 0
    Running the bandwidthTest program ensures that the system and the CUDA-capable device are able to communicate correctly. Its output is shown in
    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/graphics/valid-results-from-sample-cuda-bandwidthtest-program.png
    Note that the measurements for your CUDA-capable device description will vary from system to system. The important point is that you obtain measurements, and that the second-to-last line (in Figure 2) confirms that all necessary tests passed.Should the tests not pass, make sure you have a CUDA-capable NVIDIA GPU on your system and make sure it is properly installed.
    If you run into difficulties with the link step (such as libraries not being found), consult the Linux Release Notes found in the doc folder in the CUDA Samples directory.
 12) http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf

     Read this pdf, bookmark it as this contains all the information possible for this. 
 13) https://developer.nvidia.com/rdp/cudnn-download
     Download the corresponding version of CuDNN(requires sign in )
 14)  Refer to this for installation:
     https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
 15) tar -xzvf cudnn-9.0-linux-x64-v7.tgz
     Unzip the cuDNN package.
 16) Copy the following files into the CUDA Toolkit directory.
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn9.2 /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn 
 17) To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v7 directory in the debian file.
     Copy the cuDNN sample to a writable path.
      cp -r /usr/src/cudnn_samples_v7/ $HOME
     Go ton the writable path.
      cd  $HOME/cudnn_samples_v7/mnistCUDNN
     Compile the mnistCUDNN sample.
      make clean && make
     Run the mnistCUDNN sample.
      ./mnistCUDNN
      If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:
       Test passed!
  18) MxNet Installation can be found here:
      https://mxnet.incubator.apache.org/install/index.html
      I will give the instructions for Ubuntu using Pip
  19) sudo apt-get update
sudo apt-get install -y wget python
wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
  20) nvcc --version (ensure 9.2 is installed can skip)
  21) pip install mxnet-cu92 
  22) Run any small program importing mxnet and see if it compiles without any issue
      import mxnet as mx
       a = mx.nd.ones((2, 3), mx.gpu())
       b = a * 2 + 1
       b.asnumpy()
  23) I wouldn't reccomend Jupyter for running this code, choose to run it locally using terminal line
      Install numpy with scipy and others
        sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
      Install openCV2
         sudo apt-get install python-opencv
      install scikit learn
         pip install -U scikit-learn
      pip install easydict (https://github.com/makinacorpus/easydict) 
       In case of error such as no module called face_image or face_preprocess:
       put this line before the import statement
         sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
       In the case of izip import error, replace the izip import with
        try:
    from future_builtins import zip
except ImportError: # not 2.6+ or is 3.x
    try:
        from itertools import izip as zip # < 2.5 or 3.x
    except ImportError:
        pass


    pip install multiprocess

   24) Modules Used:
       argparse (for command line arguments)
       os (to refer to paths apart from the current directory)
       
   