#!/bin/bash
#States of citys.

echo "Choose DevOps Tools Name Chef Ansible Docker Jenkins "
read tool
case $tool in
Chef) echo "Chef is CM Tool :)"
;;
Ansible) echo "Ansible is CM Tool"
;;
Docker) echo "Docker is container tool"
;;
Jenkins) echo "Jenkins is CI/CD Tool"
;;
*) echo "I don't know"
;;
esac

