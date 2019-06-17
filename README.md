# Grab AI For SEA Safety Challenge

Skip to [Step to predict hold-out Dataset](#step-to-predict-hold-out-dataset)

# Introduction
In this reposity I consider myself exploring method of binary classifying telematics data into safe and unsafe category. The safety behavior were manually labelled by trip's passengers hence part of the data could have been said to have bias towards multipe reasons that probably unexist in the dataset such as car model, gender, time of the trip, road condition, weather and etc.

To get understanding of the telematic data, I've had voluntarily recorded my own driving data and part of the analysis and the dataset is released publicly on [Kaggle](https://www.kaggle.com/mahadir/phone-telematics-exploratory).

Also couple of literatures were reviewed related to telematics and driving behavior to get rough idea on how this type of problem is tackled, primarily I found article and video from Uber is aspiring, [here](https://eng.uber.com/telematics/) and [here](https://www.youtube.com/watch?v=_s8ZPVNKsGk).

# Modelling
I've taken some steps into exploring few models to get not only high performance model but what if we can run this model in massively distributed way like in small device such as driver mobile phone where an almost real-time prediction can be made and notify the driver right away . I found that deep learning like Tensorflow can be ported both in iOS and Android. As such,  I've looked into convolutional Neural Net. The LSTM part however is for me to answer question what if we don't have to perform padding or truncate the observation but instead let the model learn the all correlation and by using the Autoencoder we can extract a fixed length of feature and feed into secondary model.

The selection of XGboost were inspired by the Uber documentation where this way huge dataset can be preprocessed and trained in Spark.  And right now, from my experiment the most established model and high performance is xgboost, below is the list of the models with test set AUC score.

* XGboost with hand picked features (0.75)
* XGboost with raw data (test set AUC: 0.73)
* LSTM Autoencoder with XGboost (0.66)
* Convolutional Neural Net (test set AUC: 0.73) **Note: something wrong with this model [TODO: Debug]**

# Exploring and Visualization
The part I always enjoyed is exploring and finding insight in the data. Sometimes I wandered too long and spend too much on this part until too little time left on modelling. Yet this part should be iterative where after modelling we should perform post mortem to look for ways to improve the performance futher. The notebook involved here are as listed below:

* [exploration.ipynb](exploration.ipynb)

## PCA
Just last minute I performed PCA on the preprocessed with **Preprocess-hold-out Dataset.ipynb** looks like 2 PCA components preserved 66% of information and from the plot t̶h̶e̶ ̶s̶a̶f̶e̶ ̶a̶n̶d̶ ̶u̶n̶s̶a̶f̶e̶ ̶c̶a̶n̶ ̶b̶e̶ ̶s̶e̶p̶a̶r̶a̶t̶e̶d̶ ̶b̶y̶ ̶b̶o̶u̶n̶d̶a̶r̶y̶. Edit: The '0' overlapped the '1'

I include the notebook that produced the PCA but no time to run another model from this data [Xgboost raw data post-mortem analysis.ipynb](Xgboost%20raw%20data%20post-mortem%20analysis.ipynb)

<img src="/images/PCA.png" width="500" />

## Speed 
<img src="/images/safe-speed.png" width="500" />
<img src="/images/unsafe-speed.png" width="500" />

## Acceleration from speed
<img src="/images/safe-speed-acc.png" width="500" />
<img src="/images/unsafe-speed-acc.png" width="500" />

##  Gyro
<img src="/images/safe-gyro.png" width="500" />
<img src="/images/unsafe-gyro.png" width="500" />

## Accelerometer
<img src="/images/safe-acceleration.png" width="500" />
<img src="/images/unsafe-acceleration.png" width="500" />

## Bearing
<img src="/images/safe-bearing.png" width="500" />
<img src="/images/unsafe-bearing.png" width="500" />


# Step to predict hold-out Dataset

## Hold-out Dataset
It's assumed that the structure of the hold-out dataset is similar to the provided dataset where features and label is separated with different directory with partitioned CSV files. The content of both labels and features also is assumed to contain similar structures.
```
safety/features/part-PARTITION_NUM-GUID.csv
safety/labels/part-PARTITION_NUM-GUID.csv
```

## Hardware Requirements:
Depending on the size of the dataset, but my environment on training the data without much worrying about the kernel restarted due to limited I've chosen Google Cloud Platform with this type of machine n1-highmem-8 (8 vCPUs, 52 GB memory) on their datalab service. I also attached single GPU NVIDIA Tesla K80 to get boost on computing and save time.

To create the same environment instance the command as such:
```
$ datalab beta create-gpu grabaiforsea --machine-type n1-highmem-8 --accelerator-count 1
```

## Software requirements:
The codes were developed using Python 3.5.6 (Anaconda). Due to the nature of using Jupyter notebook from datalab most of the computing libraries are already installed. 

## Steps
1. Git clone this repository
2. Open **XGboost Feature engineered for holdout.ipynb** using jupyter notebook
3. Replace **path_to_feature_dir** and **path_to_label_dir** to the location of the dataset
```
path_to_feature_dir = "safety/features/"
path_to_label_dir = "safety/labels/"
```
4. Execute all cells


# TODO
I feel that there are so many areas I haven't covered yet in this dataset that I wish I could have more time diving into it. Listed here few part that I should've done or something I would revisit in future.

* The preprocessing should be a dedicated ETL code to be able to scale with Terabytes of data, at current company we're using Pyspark and airflow orchestration 
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
