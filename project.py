import pandas as pd
import numpy as np
import re, glob, sys, zipfile
from skimage import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from skimage.transform import rescale


preprocessor = sys.argv[3]    # preprocessing option
reg_pattern = 'katkam\W([\d]+)' 
csv_directory = 'yvr-weather'
image_directory = 'katkam-scaled'
OUTPUT = (
    '\nPreprocessing method: {method}\n'
    '-----------------------\n'
    'bayes model accuracy: {bayes_accuracy:.3g}\n'
    'k_neighbour model accuracy: {k_neighbour_accuracy:.3g}\n'
    'svc model accuracy: {svc_accuracy:.3g}\n'
)


#------------------------------------------------------------------------------
# Dataframe helper functions
#------------------------------------------------------------------------------
# function to print out unique items in each dataframe column
def printUniqueValueInEachColumn(df):
    headers = list(df)
    for header in headers:
        print(header + "\n")
        print(str(df[header].value_counts().index) + "\n\n")




#------------------------------------------------------------------------------
# Dataframe apply functions
#------------------------------------------------------------------------------
# function to extract date from path
def extract_date(path):
    match_reg_pattern = re.search(reg_pattern, path)
    if match_reg_pattern:
        return match_reg_pattern.group(1)



# function to extract filename from path    
def extract_filename(path):
    match_reg_pattern = re.search(reg_pattern + '.jpg', path)
    if match_reg_pattern:
        return match_reg_pattern.group(0)


# function takes an image path,
# extracts the image array,
# separates it into separate r, g, b layers
# and returns a weighted average value from each layer
def add_image0(imagepath):
    image = io.imread(imagepath)
    image_r, image_g, image_b = separate_image_layers(image)
    average_r = np.average(image_r)
    average_g = np.average(image_g)
    average_b = np.average(image_b)
    return average_r + average_g + average_b



# function takes an image path
# extracts the image array,
# scale it 1/4 size
# reshape each image to a 1D array
# and returns a weighted average value from each layer
# NOTE: to be used with PCA only
def add_image1(imagepath):
    image = io.imread(imagepath)
    image_rescaled = rescale(image, 0.25, mode='reflect') # 1/4 scale to overcome memory issues
    output = image_rescaled.reshape(1, -1) # reshape to 1D array
    return np.squeeze(output)
    


# function takes an image path
# extracts the image array,
# divides the image to random patches and takes the average value 
def add_image2(imagepath):
    image = io.imread(imagepath)
    patches = extract_patches_2d(image, (24, 32), max_patches=8, 
        random_state=np.random.RandomState(0))
    average = np.average(patches)
    return average


# function takes an image path
# extracts the image array,
# crops out the sky and returns its average 
def add_image3(imagepath):
    image = io.imread(imagepath)
    sky = image[:48,:256,:]
    average = np.average(sky)
    return average
    
    
# function takes an image path
# extracts the image array,
# crops out the sky, tree, road, sea sections and takes average
def add_image4(imagepath):
    image = io.imread(imagepath)
    sky = image[:96,:256,:]
    sky_patches = extract_patches_2d(sky, (12, 16), 
        max_patches=8, random_state=np.random.RandomState(0))
    road = image[150:,:50,:]
    trees = image[144:175,210:,:]
    sea = image[125:,100:200,:]
#    bottom = image[96:, 128:, :]
    
    average1 = np.average(sky_patches)
    #average1 = np.average(sky)
    average2 = np.average(trees)
    average3 = np.average(road)
    average4 = np.average(sea)
    #total = average1 + average2 + average3 + average4
    #return total
    return average1, average2, average3, average4   
    
    
# function to remove certainty items in weather column
def clean_description(stringlist):            
    contains = 'Heavy|Moderate|Mostly|Mainly|Showers|Pellets|Fog|Freezing|nan'
    
    # remove matching words in list    
    for word in stringlist:        
        match = re.match(contains, word)
        if match:
            stringlist.remove(word)
        
    # remove repetition or if string contains "Fog"
    while len(stringlist) > 0:
        if ( (len(stringlist) > 1) and (stringlist[0] == stringlist[-1]) ) \
            or (stringlist[-1] == 'Fog'):
            stringlist.remove(stringlist[-1])
        else:
            break
    
    output = ','.join(stringlist)
    if output == "":
        return None
#    elif output == 'Rain,Snow':
#        return 'Rain'
#    elif output == 'Thunderstorms':
#        return 'Rain'
    
    # else
    return output




#------------------------------------------------------------------------------
# Image output functions
#------------------------------------------------------------------------------

# The following two functions are adapted from 
# http://blog.yhat.com/posts/image-processing-with-scikit-image.html
def display_image(images_rgb, titles):
#    plt.clf()
    plt.figure()
    
    i = 0
    for image, title in zip(images_rgb, titles):
        plt.subplot(1, len(images_rgb), i + 1)
        plt.title(title)
        plt.axis('off')
        plt.imshow(image)
        i = i + 1
    
    plt.show()

    
def separate_image_layers(image_rgb):
    image_r, image_g, image_b = \
        image_rgb.copy(), image_rgb.copy(), image_rgb.copy()
    # switch off other color layers to show isolated r, g, b layers
    image_r[:,:,(1,2)] = 0
    image_g[:,:,(0,2)] = 0
    image_b[:,:,(0,1)] = 0
    
    return image_r, image_g, image_b



#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

def main():
    # create dataframe from csv files in zip
    with zipfile.ZipFile(sys.argv[1],"r") as csv_zip:
        csv_zip.extractall()
        
    csv_files = glob.glob(csv_directory + '/*.csv')
    
    # make sure csv directory is not empty
    if len(csv_files) <= 0:
        print("no csv files to process")
        return 
    
    dataframes = []

    for csv_file in csv_files:
        table = pd.read_csv(csv_file, sep=',', skiprows=16, parse_dates=[0])
        dataframes.append(table)
    df = pd.concat(dataframes)
#    printUniqueValueInEachColumn(df)

    
    
    
    # create dataframe from image file names
    with zipfile.ZipFile(sys.argv[2],"r") as image_zip:
        image_zip.extractall()
        
    image_files = glob.glob(image_directory + '/*.jpg')
    
    # make sure image file directory is not empty
    if len(image_files) <= 0:
        print("no image files to process")
        return 

    image_df = pd.DataFrame({'path' : image_files})
    image_df['filename'] = image_df['path'].apply(extract_filename)

    image_df['Date/Time'] = pd.to_datetime(
        image_df['path'].apply(extract_date),
        infer_datetime_format=True
    )
    
    
    
    # preclean image_df
    # get data of each image, depending on preprocessing options
    if preprocessor == '0':
        image_df['image'] = image_df['path'].apply(add_image0)
    elif preprocessor == '1':
        image_df['image'] = image_df['path'].apply(add_image1)
    elif preprocessor == '2':
        image_df['image'] = image_df['path'].apply(add_image2)
    elif preprocessor == '3':
        image_df['image'] = image_df['path'].apply(add_image3)
    elif preprocessor == '4':
        image_df['image'] = image_df['path'].apply(add_image4)
    else:
        print("Invalid preprocessing option: must be 0, 1, 2, 3, or 4")
        return
        
    
    # separate tuple/array data into columns
    image_vals = image_df['image'].apply(pd.Series)
    image_df.drop(labels=['path', 'filename', 'image'], axis=1, inplace=True)
    image_vals = image_df.merge(image_vals, how='left', left_index=True, right_index=True)


    # preclean df
    # remove NaN columns and columns consisting mostly of NaN
    df.dropna(axis=1, how='all', inplace=True)
    df = df.select(lambda x: not re.search('Quality|Chill|Hmdx', x), axis=1)
        
        
    # change column names
    df.columns = ['Date/Time', 'year', 'month', 'day', 'hour',
                  'temp', 'dew_temp', 'rel_hum', 'wind_dir',
                  'wind_speed', 'visibility', 'pressure', 'weather']


    # merge dataframes df and image_vals
    df = image_vals.merge(df, how='left', on='Date/Time')
    
    
    # clean time 
    df.drop(labels=['Date/Time'], axis=1, inplace=True)
    df['hour'] = pd.to_datetime(df['hour'], format='%H:%M').dt.hour # keep hour only
    

    # clean the weather labels
    df['weather'] = df['weather'].astype('str') 
    df['weather'].replace(to_replace='Drizzle', value='Rain', inplace=True, regex=True)
    df['weather'] = df['weather'].str.split(pat=' |,')
    df['weather'] = df['weather'].apply(clean_description)
    #df['weather'].value_counts()
        
    


    # create_ml_datasets
    weather_described = df.dropna(axis=0, how='any') #select rows without null values
    X = weather_described.drop(labels=['weather'], axis=1)
    y = weather_described['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    
    # create bayes, k-neighbor, and svc models
    bayes = make_pipeline(
        StandardScaler(), 
        GaussianNB() 
    )
    
    k_neighbour = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=8)
    )
    
    svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=5)
    )
    

    # fit models
    bayes.fit(X_train, y_train)
    k_neighbour.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    
    
    # print scores
    print(OUTPUT.format(
        method = preprocessor,
        bayes_accuracy = bayes.score(X_test,y_test),
        k_neighbour_accuracy = k_neighbour.score(X_test,y_test),
        svc_accuracy = svc.score(X_test,y_test)
    ))

    
    
if __name__ == '__main__':
    main()
