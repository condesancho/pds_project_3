# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import genfromtxt
from PIL import Image

my_data = genfromtxt('denoised_matrix.csv', delimiter=',')
im = Image.fromarray(my_data)
im.save("denoised_house.csv")

