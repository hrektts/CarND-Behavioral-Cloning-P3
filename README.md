# **Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

[//]: # (Image References)

[mse]: ./fig/mse.png "Training and validation error"
[bridge]: ./fig/bridge.png "Bridge"
[corner]: ./fig/corner.png "Corner"
[center]: ./fig/center_2016_12_01_13_31_13_037.jpg "Center image"
[left]: ./fig/left_2016_12_01_13_31_13_037.jpg "Left image"
[right]: ./fig/right_2016_12_01_13_31_13_037.jpg "Right image"

## Files Submitted

My project includes the following files:

- model.py: containing the script to create and train the model
- drive.py: for driving the car in autonomous mode
- model.h5: containing a trained convolution neural network
- README.md: summarizing the results

## Getting start driving

Using the Udacity provided simulator and my drive.py file, the car can be driven
autonomously around the track by executing

```sh
python drive.py model.h5
```

Here is a [link](https://youtu.be/JiSAV5L6wCc) to a video recorded using my model.

## Model Architecture

My model based on the NVIDIA's model[[1](#nvidia)] (model.py lines 32-46). The changes
from the base model are the insertion of the cropping layers following the
normalization layer and the dropout layers following fully connected layers
(model.py lines 41, 43, 45).

First, I inserted the cropping layers in order to preprocess imput images. The
details are descrived in the following section.

Second, I added dropout layers to reduce overfitting. I found that the base model
had a low mean squared error on the training set but a high mean squared error on
the validation set. This implied that the model was overfitting. To avoid
overfitting, I added dropout layers following each fully connected layer. As a
result of this change, I succeeded in reducing overfit as shown in the figure
below.

![Training and validation error][mse]

Finally, my model is as follows:

``` shell
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

## Model parameter

I use the following parameters to train the model:

- Validation split: 20% of the data
- Batch size: 32
- Number of epochs: 8
- Optimizer: Adam

The model used an adam optimizer, so the learning rate was not tuned manually.

## Training Strategy

To find the solution, I took an iterative approach. First, I traind the base
model with the data set, that provided by Udacity. However, the car driven by
the model did not work well. Then I added modified data and newly collected
data. Consequently, I successfully created my model.

In the whole step of my trial and error, the images were cropped to remove the
background on the upper side and the hood of the car on the lower side, and
normalized.

In the first trial, I used the model descrived above and the data set provided
by Udacity. The driver based on the trained model tended to go to the left.
I thought that this might be due to the fact that the provided data set contains
many left curves than right curves.

In the next step, I added flipped images to the original data set
(model.py lines 104-106). The model trained by the augmented data set could run
the car straight on the simulator. However the car stucked on the bridge at the
middle of the course.

![Bridge][bridge]

I thought that the model could not trained well for crossing the bridge because
the number of the images that captured the scenery of the bridge was relativery
small than one of the images that captured other part of the course.

In the next step, I added some images that were recorded the scenes of crossing
the bridge for the data set. The model trained by the augmented data set
successfully run the car on the bridge. However, it failed to turn around a
corner following the bridge, and went straight.

![Corner][corner]

Certainly there was a way to go straight, but I wanted the car to turn to the
left.

To fix this, I added some images that were recorded the scenes of passing the
corner for the data set. The model trained by the augmented data set
successfully passed the corner.

Finally, I added some images to the data set to make the car run smoothly.
The simulator run in training mode could capture images that were took from
different positions as shown below.

|Left image   |Center image     |Right image    |
|-------------|-----------------|---------------|
|![Left][left]|![Center][center]|![Right][right]|

In this step, I used whole three kind of images for training my model though
I used only the center image up to the previous step. Concretely, I added the
left images along with the steering angle that is added 0.05 to the original
angle (model.py lines 92-96), and the right images along with the steering
angle that is subtracted 0.05 from the original angle (model.py lines 98-102).
These mean that the car should go right when it on the left side of the road,
and should go left when it on the right side of the road. As a result, the
car could run smoothly on the track 1 of the simulator.

## Reference

- <a name="nvidia">[1]
  Firner, Ben et al. “End-to-End Deep Learning for Self-Driving Cars.”
  Parallel Forall, Nvidia, 25 Aug. 2016,
  devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/.
  Accessed 12 Mar. 2017.
