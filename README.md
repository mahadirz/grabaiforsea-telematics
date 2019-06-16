# Grab AI For SEA Safety Challenge

TL;DR skip to [Step to execute hold-on Dataset](#step-to-execute-hold-on-dataset)

# Introduction
In this reposity I consider myself exploring method of binary classifying driving behavior into safe and unsafe. The safety behavior were manually labelled by trip's passengers hence part of the data could have been said to have bias towards multipe reasons that probably unexist in the dataset such as car model, gender, time of the trip, road condition, weather and etc.

To get understanding of the telematic data, I've had voluntarily recorded my own driving data and part of the analysis and the dataset is released publicly on [Kaggle](_).

Also couple of literatures were reviewed related to telematics and driving behavior to get rough idea on how this type of problem is tackled, primarily I found article and video from Uber is aspiring, [here](https://eng.uber.com/telematics/) and [here](https://www.youtube.com/watch?v=_s8ZPVNKsGk).

# Modelling
I've taken some steps into exploring few models to get not only high performance model but what if we can run this model in massively distributed way like in small device such as driver mobile phone where an almost real-time prediction can be made and notify the driver right away . I found that deep learning like Tensorflow can be ported both in iOS and Android. As such,  I've looked into convolutional Neural Net. The LSTM part however is for me to answer question what if we don't have to perform padding or truncate the observation but instead let the model learn the all correlation and by using the Autoencoder we can extract a fixed length of feature and feed into secondary model.

The selection of XGboost is inspired by the Uber method where this way huge dataset can be preprocessed and trained in Spark.  And right now, from my experiment the most established model and high performance is xgboost, below is the list of the models.

* XGboost with raw data (test set AUC: 0.73)
* XGboost with hand picked features ()
* LSTM Autoencoder with XGboost ()
* Convolutional Neural Net (test set AUC: 0.73)

# Exploring and Visualization
The part I always enjoyed is exploring and finding insight in the data. Sometimes I wandered too long and spend too much on this part until too little time left on modelling. Yet this part should be iterative where after modelling we should perform post mortem to look for ways to improve the performance futher. The notebook involved here are as listed below:

* 1
* 2

# Step to execute hold-on Dataset

## 
* It's assumed that the structure of the hold-on dataset have similar structure as the provided dataset.

## Machine Requirements:
Depending on the size of the dataset, but my environment on training the data without much worrying about the kernel restarted due to limited I've chosen Google Cloud Platform with this type of machine n1-highmem-8 (8 vCPUs, 52 GB memory) on their datalab service. I also attached single GPU NVIDIA Tesla K80 to get boost on computing and save time.

To create the same environment instance the command as such:
`datalab beta create-gpu grabaiforsea --machine-type n1-highmem-8 --accelerator-count 1`


# TODO
I feel that there are so many areas I haven't covered yet in this dataset that I wish I could have more time diving into it. Listed here few part that I should've done or something I would revisit in future.

* The preprocessing should be a dedicated ETL code to be able to scale with Terabytes of data, at my current company we're using Pyspark and airflow orchestration 
* The label class is imbalanced, I probably should've upscale it
* Tune the model to be more sensitive toward false positive (perhaps)
* Heatmap of v-a and execute ML on the heatmap or the PCA
* I still believe in the LSTM Autoencoder and the ConvNet, should spend more time into it
* Write a mobile app with the model embedded and notify in real-time


# References
1. Feature extraction from telematics car driving heatmaps (with G. Gao). European Actuarial Journal 8/2 (2018)	
2. Convolutional neural network classification of telematics car driving data 
(with G. Gao). Risks 7/1 (2019)
3. [Machine Learning with Spark: Kaggle’s Driver Telematics Competition](https://dzone.com/articles/machine-learning-with-spark-kaggles-driver-telemat)
4. [Uber Engineering](https://eng.uber.com/telematics/)