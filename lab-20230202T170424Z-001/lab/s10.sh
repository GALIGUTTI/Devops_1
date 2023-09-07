#!/bin/bash
while getopts ":eabcd" options;
do
case $options in
a) echo "received -a";;
b) echo received -b;;
c) echo received -c;;
d) echo received -d;;
*) echo Unknown value received;;
esac
done
