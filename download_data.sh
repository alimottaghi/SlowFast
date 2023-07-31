#!/bin/bash

cd /data
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

cd /data/domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt


cd /data
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