#!/bin/bash

rm -r build
mkdir build
cd build
source /opt/crosstools/mobilint/1.0.0/v3.2.1/environment-setup-cortexa53-mobilint-linux
cmake ..
make -j8
cd ..