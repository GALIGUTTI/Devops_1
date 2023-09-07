#!/bin/bash
echo "select traffic signals RED GREEN YELLOW"
read colour
if [ $colour == RED ];
then
echo "Red colour indicates STOP"
elif [ $colour == GREEN ];
then
echo "GREEN colour indicates START"
elif [ $colour == YELLOW ];
then
echo "Yellow colour indicates ready to start"
else
echo "Sorry This colour is not avalible in our script!"
echo "Thank you!!.."
fi
