# pds_project_3

Students involved : </br>
                    Arailopoulos Vasilis [github_profile](https://github.com/condesancho)  email : arailosb@gmail.com</br>
                    Filis Charis          [github_profile](https://github.com/harryfilis)   email : harry.filis@yahoo.gr</br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/condesancho/pds_project_3/blob/master/LICENSE)
#
## Non-Local Means Filter – Accelerated with CUDA.
## Usage 
prerequisites:
--nvcc compiler with cl.exe </br>
--gcc compiler</br>
You can run the code in google colab as well and we applaud this.

To use the code first you need to compile it.
We have made a Makefile so you can go to Makefile directory and type make in terminal.

Run:
```sh
  ./executable [Path to file] [No. image rows] [No. image cols] [Patch size]
  (or for random image matrices)
  ./executable [Path to file] [No. image rows] [No. image cols] [Patch size]
```

Create csv for Image(if rgb if not just delete the line rgb2gray and do the same)</br>
run on Matlab the funticion JpgToCSV("PATH")</br>
put noise by running ΑpplyGaussianNoise.m</br>
execute codes with the noisyfile</br>
run ReadAndPrintImage.m</br>
## Abstract

Non-Local Means is an algorithm applied to images in order to denoise them.

First of all the Filter demands to calculate some weights with which the calculation of the weighted average is done.

![equation](https://i.imgur.com/Exs82hi.png)

Where N_k is a neighborhood with the adjoined pixels to pixel k.This equitation gets the gaussian filtered patches of the image.

The parameter Z(i) is computed by the equation


![equation2](https://i.imgur.com/QYgRDqJ.png)


Long story short the updated value of the pixel(assuming that we have a grayscale image and each pixel gets a value ex. 0-255) Is a result of the weighted average shown below.

![equation3](https://i.imgur.com/QtX3sYe.png)

Where Ω is the definition space of the picture f:Ω -> R the initial picture with noise and f_hat the approximation of the denoised image.

In this assignment the goal is to accelerate the algorithm using CUDA and turn the initial complexity of _O(N^4)_ to _O(N^2)_.

## Results
![lena](https://github.com/condesancho/pds_project_3/blob/master/DenoisedImagesPictures/Lena_256.png)
![umbrella](https://github.com/condesancho/pds_project_3/blob/master/DenoisedImagesPictures/GirlWithUmbrella.png)

The perfect application for this filter though is in biomedical pictures for example magnetic resonance picture
![mag](https://github.com/condesancho/pds_project_3/blob/master/DenoisedImagesPictures/magnetic_resonance_256.png)
