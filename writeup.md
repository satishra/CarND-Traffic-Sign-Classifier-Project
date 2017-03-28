#**Traffic Sign Recognition** 

---
**Traffic Sign Recognition Project**

The goals / steps follwed for this project:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
####1. Base line of my project implementation is LeNet.
You're reading it! and here is a link to my [project code](https://github.com/satishra/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Traing data was taken from https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

The code for this step is contained in the first[1] code cell of the IPython notebook.  

I used the pickle library to extract data 
signs data set:

* The size of training set is : 34799 
* The size of test set is : 12630 
* The shape of a traffic sign image is : 32x32x3
* The number of unique classes/labels in the data set is : 43

####2. Randomly selected a sample from the training set and validated the labeling.

The code for this step is contained in the [2] code cell of the IPython notebook.  

###Design and Test a Model Architecture

####1. As the first step data is preprossed, converted the training set to yuv using CV libraries.

The code for this step is contained in the [4] code cell of the IPython notebook.

As a first step, I decided to convert the images to yuv because after experimentation I got better validation accuracy, hence I am keeping yuv conversion of the training data.

####2. For the training I decided to shuffle the data using sklearn library and use batch of 128 samples for the training.

The code for splitting the data into training and validation sets is contained in the 12 code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by shuffle function imported from sklearn utils.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Model is mentioned below

The code for my final model is located in the 8 cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| 2d Convolution      	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| 2d Convolution      	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten 	    | input 5x5x16, output=400     									|
| Fully connected		| input 400, output=120    									|
| RELU					|												|
| Fully connected		| input 120, output=84     									|
| RELU					|												|
| Fully connected		| input 84, output=43     									|
 
####4. Please find the traing model followed below

The code for training the model is located in the [10] [11] and [12] cell of the ipython notebook. 

To train the model, I used an adamoptimer from tensor flow.While training I tuned the learning rate and EPOCHS.
I was checking the change in accuracy to limit the number of EPOCHS need to be used.
Max EPOCHS is set to 100 but I am terminating training before Epochs reaches 100 if the validation accuracy is same for n(5) consecutive times.
I am also checking that the validation accuracy is greater than 0.90 before terminating training.
On an average only about 25-50 Epochs are used for training.

####5. Final accuracy numeber are mentioned below.

The code for calculating the accuracy of the model is located in the 13 cell of the Ipython notebook.

My final model results were:
* training set accuracy of : 98.4
* validation set accuracy of : 92.3
* test set accuracy of : 92.3

###Test a Model on New Images

####1. Please find the my observations with test images took from German traffic signs.

I have taken 12 German traffic signs that I found on the web and Accuracy of detection is 100%.
New images used for testing are in https://github.com/satishra/CarND-Traffic-Sign-Classifier-Project/tree/master/new_images
Code and its results can be found in [13] cell.

####2.Top 5 Softmax output.
Printed out the top 5 softmax outputs in [15] code cell, With Probablity values close to 1/0.99 for the correctly predicted images.

Final conclusion:
New images are predicted correctly with the trained data test accuracy data.
Analyzing the data it looks like I have to keep the overfitting in control. For ex, one of the sign was wrongly identified during one of the iterations but improving the overfit condition solved this issue.
