#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys


def reName(filpath):
    image_list = os.listdir(filpath)
    for image_index in range(0, len(image_list)):
        full_name = os.path.join(filpath, image_list[image_index])
        image_name = image_list[image_index]
        image_name = image_name.split(".")[0]
        rename = image_name[:-5]
        rename = rename + ".nii.gz"
        os.rename(full_name, os.path.join(filpath, rename))


if __name__ == '__main__':
    #filepath = sys.argv[1]
    filepath = r"C:\Users\wlc\PycharmProjects\pythonProject\data\synapse\imagesTr"
    reName(filepath)
