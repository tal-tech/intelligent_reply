### 环境配置

FROM registry.cn-beijing.aliyuncs.com/ailab-paas/cuda10.1-cudnn7-ocv4.2.0-torch1.5.0-py3.7-ubuntu18.04:v1.0.2


RUN python3.7 -m pip install futures==3.1.1 torch==1.5.0+cu101 torchvision==0.6.0+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    --timeout 1800 \
    -i https://mirrors.aliyun.com/pypi/simple/


RUN apt-get install -y pkg-config && python3.7 -m pip install --upgrade pip

COPY ./requirements.txt /tmp/requirements.txt
RUN python3.7 -m pip install --timeout 1800 -r /tmp/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
