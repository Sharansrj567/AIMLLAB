# The Z-score(also called the standard score) is an important concept in statistics that indicates 
# how far away a certain point is from the mean. 
# By applying Z-transformation we shift the distribution and make it 0 mean with unit standard deviation. 
# For example — A Z-score of 2 would mean the data point is 2 standard deviation away from the mean.
# Z-score(i) = (x(i) -mean) / standard deviation
# It assumes that the data is normally distributed and hence the % of data points that lie between -/+1 stdev. is ~68%,
# for -/+2 stdev. is ~95% and -/+3 stdev. is ~99.7%. 
# Hence, if the Z-score is >3 we can safely mark that point to be an outlier.


import numpy as np
data = [1, 2, 3, 2, 1, 100, 1, 2, 3, 2, 1]
threshold = 3
mean = np.mean(data)
std = np.std(data)
z_score_outlier = [i for i in data if (i-mean)/std > threshold]
print (z_score_outlier)