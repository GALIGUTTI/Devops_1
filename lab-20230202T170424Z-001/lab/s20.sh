#!/bin/bash
i=5
until [ $i -gt 15 ]
do
echo "number $i"
i=$(( i+1 ))
done
