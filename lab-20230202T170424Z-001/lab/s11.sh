#!/bin/bash
echo "Enter your Age:"
read age
if [ $age -ge 18 ]
then
echo "your eligible for voting"
else
echo "your not eligible for voting"
echo "Thank You!."
fi
