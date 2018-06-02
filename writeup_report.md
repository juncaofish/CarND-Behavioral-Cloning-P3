# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.png "Model Visualization"
[image2]: ./examples/center_line.png "Center Line"
[image3]: ./examples/cropped_image.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results  
* video.mp4 - A video recording of your vehicle driving autonomously at least one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for loading collected images and steering angle data, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture is borrowed from the aforementioned NVIDIA paper in which they tackle the same problem of steering angle prediction, and is shown as below:
![alt text][image1]

Input normalization is implemented through a Lambda layer, which constitutes the first layer of the model. In this way input is standardized such that lie in the range [-1, 1]: of course this works as long as the frame fed to the network is in range [0, 255].

Convolutional layers are followed by 3 fully-connected layers: finally, a last single neuron tries to regress the correct steering value from the features it receives from the previous layers.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 73,75,77,79,81). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 107). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it performs greet for minist classification task.

My second try was to use the network in Nvidia paper.

For both networks, the car complete full laps without running outside the track. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model needed more epochs to reach a low mean squared error on the validation set. And the mean squared error on validation set was higher than the training set after some epochs. This implied that the model was overfitting. 

The Nvidia network reaches a lower loss on both training and validation data set after only 3 epochs.
So finally, I use this network to traing a model for driving.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I take more attention when teaching the car to drive around these spots to collect more precise steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps(one in clockwise, the other in counter-clockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer both left and right direction.

After the collection process, I had 5501 number of data points. Then I use images collected by both left and right cameras to get 3 times data. This step also helps teaching the car how to steer back to the center. Then I preprocessed this data by cropping the image to exclude portion of the sky and the hood of car (model.py line 48).
![alt text][image3]

I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the growing epochs will only reduce loss on training set but the validation loss doesn't change much. I used an adam optimizer so that manually training the learning rate wasn't necessary.
