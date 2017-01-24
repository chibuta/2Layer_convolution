# Convolution neural network

This is a tensorflow implentation of deep layer convolution neural network on the [ORL faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).

<img src="faces.gif" width="400" height="400" />

#Runing the script

Unzip the dataset first 
```sh
$ python convnet.py
```
A successful run will should display a plot of the loss function and accuracy as shown below..

<img src="loss.jpg" width="800" height="350" />


#Visualisation of trained convolution filters


Layer 1 filters:

<img src="layer1_filters.png" width="500" height="300" />



Layer 2 filters:

<img src="layer1_1filters.png" width="500" height="300" />
