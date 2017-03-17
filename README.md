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
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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

My model based on the NVIDIA's model[^nvidia] (model.py lines 32-46). The changes
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

[^nvidia]: Firner, Ben et al. “End-to-End Deep Learning for Self-Driving Cars.”
Parallel Forall, Nvidia, 25 Aug. 2016,
devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/.
Accessed 12 Mar. 2017.

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

In the next step, I added flipped images to the original data set. The model
trained by the augmented data set could run the car straight on the simulator.
However the car stucked on the bridge at the middle of the course.

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

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
