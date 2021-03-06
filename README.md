# Webcam Weather Prediction

This repository contains several prediction models for the Webcam Weather 
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

[Katkam Images](https://courses.cs.sfu.ca/2017su-cmpt-318-d1/pages/ProjectWeatherKatkam),
[Weather Data](https://courses.cs.sfu.ca/2017su-cmpt-318-d1/pages/ProjectWeatherWeather)

---

## Running the code in Python
In terminal or bash, type in

    python3 project.py weather.zip katkam-secret-location.zip PREPROCESSOR_OPTION

where the PREPROCESSOR_OPTION can be between 0 and 5.

Please take into account the time required to execute the command with
preprocessor options 1 or 4.

Terminal will output the accuracy of each prediction model on the screen.  For 
instance, below is a sample terminal output when PREPROCESSOR_OPTION = 0

    Preprocessing method: 0
    -----------------------
    bayes model accuracy: 0.69
    k_neighbour model accuracy: 0.78
    svc model accuracy: 0.836


Alternatively a bash script `script.sh` is created to batch test the code and
save you time on typing repetitive commands on terminal.  This option comes in 
handy when for some reason you want to iterate all preprocessor options in
the python3 script.  To run the bash script, in terminal or bash, type in

    ./script.sh
