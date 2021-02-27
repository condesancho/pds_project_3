# pds_project_3




#
## Non-Local Means Filter – Accelerated with CUDA.

## Abstract

Non-Local Means is an algorithm applied to images in order to denoise them.

First of all the Filter demands to calculate some weights with which the calculation of the weighted average is done.

Where is a neighborhood with the adjoined pixels to pixel k.This equitation gets the gaussian filtered patches of the image.

The parameter Z(i) is computed by the equation

Long story short the updated value of the pixel(assuming that we have a grayscale image and each pixel gets a value ex. 0-255) Is a result of the weighted average shown below.

![](RackMultipart20210227-4-fr6ci7_html_6db1d0392db7926f.png)

Where Ω is the definition space of the picture the initial picture with noise and the approximation of the denoised image.

In this assignment the goal is to accelerate the algorithm using CUDA and turn the initial complexity of _O(_to _O(_.
