#!/bin/bash

mkdir ./Build
cd ./Build

cmake ../
make -j8

cd Examples
./VogelsAbbottNet --simtime 1.0
