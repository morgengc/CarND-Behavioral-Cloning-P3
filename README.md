# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/arch.png "Architechture"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `README.md` summarizing the results
* `run.mp4` demonstrating a round of autonomous driving 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use NVIDIA DAVE-2 system, which consists of a convolution neural network (`model.py` lines 72-86).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`model.py` lines 82, 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 67-70). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the dataset which is provided by this course named `data.zip`. This dataset includes clockwise and anticlockwise driving images, within three cameras produced center, left and right images. There are more images at the turing of the road, in order to make the model smart.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to study captured images and angles within a CNN model, and predict every step of angle while in autonomous mode.

My first step was to use a convolution neural network model similar to the NVIDIA DAVE-2 system. I thought this model might be appropriate because it's widely used.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model, added some dropout layer into the model, and choosed `EPOCH=3`. The training process showed that there is no overfitting after these corrections.

Then I cropped the iamge using `Cropping2D(cropping=((70, 25), (0, 0)))` to drop the useless image data, this made the model training more faster.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at the turing of the road. To improve the driving behavior in these cases, I used a small speed in `drive.py` like `9`, and try to adjust the model architechture, remove some layer and add some dropout layer. And I set `samples_per_epoch` to 32000, which gained a great improvement.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 72-86) consisted of a convolution neural network with the following layers and layer sizes:

```
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
```

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I know I should create a good training set. However, it's not easy to play a racing game using keyboard. This process costs a lat of time, and hense the course provides a dataset, I decided to use this dataset.

I found that this dataset includes clockwise and anticlockwise driving images, within three cameras produced center, left and right images. There are more images at the turing of the road, in order to make the model smart.

first recorded two laps on track one using center lane driving, and drive reverse the lane to get more data.


I had 24108 number of data points, 1/3 of which are from center camera, 1/3 from left camera and 1/3 from right camera. I then preprocessed this data by random choose at any timestamp, and made an angle adjustment for left and right camera. Also I random choose some images to horizontally flipped.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by `val_loss` stop decrease. I used an adam optimizer so that manually training the learning rate wasn't necessary.
