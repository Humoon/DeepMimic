FROM ubuntu:20.04

#制作者信息
MAINTAINER guozongren

RUN printf "deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\n \
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\n \
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\n \
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\n \
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse\n \
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\n \
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\n \
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\n \
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\n \
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse\n" > /etc/apt/sources.list

RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
&& apt-get clean \
&& apt-get update

#设置工作路径&&安装python

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.6 python3-pip python3.6-dev
RUN  ln -s /usr/bin/python3.6 /usr/bin/python

COPY . /code
WORKDIR /code

#安装各种依赖
RUN apt-get install cmake -y
RUN apt-get install make -y
RUN apt-get update
RUN apt-get install libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev -y
RUN apt-get install build-essential -y
RUN apt-get install libglu1-mesa-dev freeglut3-dev -y

RUN apt-get install mesa-utils -y
RUN apt-get install clang -y

RUN cd tmp/swig-4.0.0 && ./configure --without-pcre
RUN cd tmp/swig-4.0.0 && make
RUN cd tmp/swig-4.0.0 && make install
RUN cd tmp/eigen-3.3.7 && mkdir build && cd build && cmake .. && make install
RUN cd tmp/freeglut-3.0.0 && cmake . && make && make install
RUN cd tmp/glew-2.2.0 && make && make install && make clean
RUN cd tmp/bullet3-3.17 && echo * && sh build_cmake_pybullet_double.sh
RUN cd tmp/bullet3-3.17/build_cmake && echo *  && make install

RUN apt install libopenmpi-dev -y
RUN apt-get update && apt-get install python-pip -y

#安装python库
COPY requirements.txt  requirements.txt
RUN python -m pip install --upgrade --force pip -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install setuptools==33.1.1 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install --upgrade pip -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install -r requirements.txt   --default-timeout=2000  -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

#更正路径防止运行报错
RUN ln -s /usr/lib64/libGLEW.so.2.1.0 /usr/lib/libGLEW.so.2.1.0
RUN ln -s /usr/lib64/libGLEW.so.2.1 /usr/lib/libGLEW.so.2.1

RUN apt install ssh -y