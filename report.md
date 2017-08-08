# Webcam Weather Prediction Report
---
 Last updated: 8/7/2017    



## Problem
This report explores using a stationary webcam, a machine learning model built 
from historical weather data, and an archive of past webcam images to make 
a correct description of the weather in real time.    


## Processing the data
Several machine learning models are created from past Kat Kam images and weather
data collected by Environment Canada at the YVR airport.  The latter is found to
contain entries with wordy or redundant weather descriptions which need
to be cleaned. Here is a list of unique weather descriptions extracted from the 
custom function call `printValueCountsInEachColumn()`:    
```
        Index(['Rain', 'Cloudy', 'Mostly Cloudy', 'Mainly Clear', 'Clear',
            'Rain Showers', 'Snow', 'Rain,Fog', 'Moderate Rain',
            'Moderate Rain,Fog', 'Fog', 'Drizzle', 'Drizzle,Fog', 'Rain,Snow',
            'Snow Showers', 'Rain,Drizzle,Fog', 'Rain,Drizzle', 'Snow,Fog',
            'Moderate Snow', 'Freezing Rain,Fog', 'Rain Showers,Fog',
            'Freezing Fog', 'Heavy Rain,Fog', 'Thunderstorms', 'Rain,Snow,Fog',
            'Moderate Rain Showers', 'Moderate Rain,Drizzle',
            'Rain Showers,Snow Showers', 'Heavy Rain', 'Moderate Rain Showers,Fog',
            'Rain Showers,Snow Pellets', 'Ice Pellets',
            'Rain Showers,Snow Showers,Fog', 'Thunderstorms,Rain Showers',
            'Snow,Ice Pellets,Fog', 'Moderate Snow,Fog', 'Rain,Ice Pellets'],
            dtype='object')
```
The custom function `clean_description()` removes adjectives `Heavy`, `Moderate`, 
`Mostly`, `Mainly` and ambiguous words such as `Showers`, `Pellets`, `Fog`, 
`Freezing` from the set of possible descriptors.

A dataframe is also created from pixel data and datetime information contained 
in the image filenames.  Due to memory issues, it is impossible to bring in every
pixel data as an independent entry the dataframe.  As a result, six workaround 
solutions are proposed:    

1. Create an equally weighted average rating of an image from 
red, green, and blue layers using `add_image0()`

    ![Figure 1](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig1.png)
 
    **Figure 1** *`add_image0()` operation*    

2. Rescale the image to 1/4 scale and reshape the image array
to 1D, and every element is brought in as an independent entry in the dataframe
using `add_image1()` 

    ![Figure 2](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig2.png)

    **Figure 2** *`add_image1()` operation*    

3. Divide the image to eight random patches and take an
equally weighted scalar average value using `add_image2()` 

    ![Figure 3](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig3.png)

    **Figure 3** *`add_image2()` operation*    

4. Create a 1D array from a sky subimage using `add_image3()`

    ![Figure 4](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig4.png)

    **Figure 4** *`add_image3()` operation*    

5. Create an equally weighted scalar average from the cropped sky subimage 
using `add_image4()`

    ![Figure 5](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig5.png)

    **Figure 5** *`add_image4()` operation*    

6. Create a tuple of scalar average ratings from cropped sky, tree, 
road, and sea subimages using `add_image5()` 

    ![Figure 6](https://csil-git1.cs.surrey.sfu.ca/byronc/webcam-weather-prediction/blob/master/fig6.png)

    **Figure 6** *`add_image5()` operation*    


## Classifiers and accuracy scores
The training set consists of approximately 2000 entries.  Naïve Bayes, KNN, and 
SVM classifiers are chosen for their speed and compatability with small data 
sets.  Methods `add_image1()` and `add_image4()` use PCA reduction to speed up
classification speeds.  The accuracy scores are shown below:


Processing method | Classifier | Accuracy Score 
----------------- | ---------- | --------------
`add_image0()` | Naïve Bayes | 0.69
  | k-neighbour | 0.78 
  | SVN | 0.836 
`add_image1()`\* | Naïve Bayes | 0.508
  | k-neighbour | 0.758 
  | SVN | 0.888
`add_image2()` | Naïve Bayes | 0.69
  | k-neighbour | 0.786 
  | SVN | 0.835
`add_image3()` | 0.708
  | k-neighbour | 0.802 
  | SVN | 0.851
`add_image4()`\* | 0.221
  | k-neighbour | 0.698
  | SVN | 0.694
`add_image5()` | 0.71
  | k-neighbour | 0.796
  | SVN | 0.874

*\* methods with PCA applied*    
 **Table 1** *Results using naïve Bayes classifier*   




## Post-mortem
Problems: 
Memory and speed issues: images have to be kept at 1/4 scale (64 x 48 pixels) before
applying PCA reduction.

Limitations:

To try:
LAB instead of RGB
