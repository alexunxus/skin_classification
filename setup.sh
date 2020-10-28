CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-10.1/" \
HOROVOD_NCCL_INCLUDE="/usr/include/" \
HOROVOD_NCCL_LIB="/usr/lib/x86_64-linux-gnu/" \
HOROVOD_NCCL_HOME="" \
HOROVOD_GPU_ALLREDUCE=NCCL \
HOROVOD_WITH_PYTORCH=0 \
HOROVOD_WITH_TENSORFLOW=1 \
pip3 install horovod --upgrade --force-reinstall
