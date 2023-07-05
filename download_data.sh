#!/bin/bash

cd /data/slowfast
mkdir -p domainnet
cd domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip

unzip clipart.zip
unzip infograph.zip
unzip painting.zip
unzip quickdraw.zip
unzip real.zip
unzip sketch.zip

rm clipart.zip
rm infograph.zip
rm painting.zip
rm quickdraw.zip
rm real.zip
rm sketch.zip

cd /data/slowfast
mkdir -p visda
cd visda
wget http://csr.bu.edu/ftp/visda17/clf/train.tar
wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
wget http://csr.bu.edu/ftp/visda17/clf/test.tar

tar xvf train.tar
tar xvf validation.tar
tar xvf test.tar

rm train.tar
rm validation.tar
rm test.tar
