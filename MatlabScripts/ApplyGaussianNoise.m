clc
clear all
close all

  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
pathToImage = './Image.csv';
I = csvread(pathToImage);
fprintf(' - applying noise...\n')
J = imnoise( I, noiseParams{:} );
imagesc(J); axis image;
colormap gray;
csvwrite('NoisyImage.csv', J);
