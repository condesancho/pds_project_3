function JpgToCSV(filename)
image1=imread(filename);
image=rgb2gray(image1);
image = mat2gray(image,[0 255])
csvwrite('Image.csv',image) 
end
