# Webcam Weather Prediction
---


This submission contains several prediction models for the Webcam Weather 
Prediction assignment.



---

## Platform, required libraries and files
The Python code is tested in Ubuntu 16.04 but theoretically should also work in 
Windows and MacOS.  You also need to have the following libraries and packages
installed:

* `python3`
* `pip`
* `Pandas`
* `numpy`
* `matplotlib`
* `scikit-image`
* `scikit-learn`

You will also have to download the katkam images and Environment Canada historical
weather data from the following links:

[Katkam Images](https://courses.cs.sfu.ca/2017su-cmpt-318-d1/pages/ProjectWeatherKatkam)
[Weather Data](https://courses.cs.sfu.ca/2017su-cmpt-318-d1/pages/ProjectWeatherWeather)

---

## Running the code in Python
In terminal or bash, type in

    python3 project.py WEATHER-CSV-FILE-DIRECTORY KATKAM-IMAGE-DIRECTORY PREPROCESSOR_OPTION

where WEATHER-CSV-FILE-DIRECTORY and KATKAM-IMAGE-DIRECTORY are the path
of your folders containing unzipped weather data and the katkam images, and
PREPROCESSOR_OPTION can be 0, 1, 2, 3, or 4.

The accuracy of each prediction model is printed on the console.
