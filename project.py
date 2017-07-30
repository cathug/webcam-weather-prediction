import pandas as pd
import numpy as np
import re, glob
from skimage import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


csv_directory = sys.argv[1]  # folder
image_directory = sys.argv[2] # image path directory
reg_pattern = 'katkam\W([\d]+)' 


# function to check list of columns in dataframe 
def printValueCountsInEachColumn(df):
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



def add_image(imagepath):
    image = io.imread(imagepath)
    #titles = "red layer", "green layer", "blue layer"
    image_r, image_g, _ = separate_image_layers(image)
    #display_image(image_rgb_layers,titles)
    average = np.average([image_r, image_g])
    #return image_r
    return average
    
    
# remove certainty items in weather column
def clean_description(string):    
    if str(string) == 'nan':
        return None
    
    words = str(string).replace(' ', ',').split(sep=',')
#    if (len(words) == 1) & (words[0] == 'Drizzle'):
#        words[0] = 'Rain'
 #   if (len(words) > 1) & (words[1] == 'Drizzle'):
 #       words.remove('Drizzle') 
    
    for word in words:
        
        for m in re.findall('Heavy|Moderate|Mostly|Mainly|Showers|Pellets', word):
            if m:
                words.remove(m)
    return ','.join(words)
#------------------------------------------------------------------------------



# clean data in dataframe
def cleanData(df):
    # change column names
    df.columns = ['path', 'filename', 'date', 'temp', 'dew_temp', 
        'rel_hum', 'wind_dir','wind_speed',
        'visibility','pressure', 'hmdx','weather']
    
    # replace NaN instances with none in weather column
    df['weather'].fillna('None', inplace=True)
    
    # list of weather descriptions
    weather_desc_list = df['weather'].value_counts().index

    # split them to individual words
    desc = np.unique(
        np.concatenate(weather_desc_list.str.split('[, ]') ) 
    )
    
    
    atmos = np.array([desc[0], desc[1], desc[3], desc[4]])
    precp = np.array([desc[10], desc[11], desc[12], desc[13], desc[14]])
    quantity = np.array([desc[2], desc[5], desc[7], desc[8], desc[6]])
    print(atmos)
    print(precp)
    print(amount)
    
    return df#, atmos, precp, quantity







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
    image.copy(), image.copy(), image.copy()
    # switch off other color layers to show isolated r, g, b layers
    image_r[:,:,(1,2)] = 0
    image_g[:,:,(0,2)] = 0
    image_b[:,:,(0,1)] = 0
    
    return image_r, image_g, image_b



def main():

    # create dataframe from csv files
    csv_files = glob.glob(csv_directory + '/*.csv')
    dataframes = []

    for csv_file in csv_files:
        table = pd.read_csv(csv_file, sep=',', 
                            skiprows=16, parse_dates=[0])
        dataframes.append(table)
    df = pd.concat(dataframes)
#    printValueCountsInEachColumn(df)


    
    # create dataframe from image file names
    image_files = glob.glob(image_directory + '/*.jpg')

    image_df = pd.DataFrame({'path' : image_files})
    image_df['filename'] = image_df['path'].apply(extract_filename)

    image_df['Date/Time'] = pd.to_datetime(
        image_df['path'].apply(extract_date),
        infer_datetime_format=True
    )


    #filter out empty columns and clean data
    # according to printValueCountsInEachColumn 
    # columns are null, except Hmdx column which has 103 non-null items
    df = df.select(
        lambda x: not re.search(
            'Quality|Flag|Year|Month|Day|^Time|Chill|Hmdx', x), 
        axis=1
    )

    df = image_df.merge(df, how='left', on='Date/Time')
    df, atmos, precp, quantity = cleanData(df)
        
    df['weather_desc'] = df['weather_desc'].apply(clean_description)


    # machine learning shit here
    weather_described = df[(df['weather'].notnull()) & 
                       (df['temp'].notnull()) &
                       (df['dew_temp'].notnull()) &
                       (df['rel_hum'].notnull()) &
                       (df['wind_speed'].notnull()) &
                       (df['wind_dir'].notnull()) &
                       (df['visibility'].notnull()) &
                       (df['pressure'].notnull()) ]

    X = weather_described.loc[:, weather_described.columns.isin(
        ['path', 'date', 'temp', 'dew_temp', 'rel_hum', 'wind_dir', 
         'wind_speed', 'visibility', 'pressure']
    )]

    X['image'] = X['path'].apply(add_image)
    X.drop(['path', 'date', 'temp'], axis=1, inplace=True)
    y = weather_described['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    bayes = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )

    bayes.fit(X_train, y_train)
    print(bayes.score(X_test,y_test))


    svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10)
    )
    svc.fit(X_train, y_train)
    print(svc.score(X_test,y_test))
    
    
    k_neighbour = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
    )

    k_neighbour.fit(X_train, y_train)
    k_neighbour.score(X_test,y_test)


#    image = io.imread(df['path'].iloc[5036])
#    titles = "red layer", "green layer", "blue layer"

#    image_rgb_layers = separate_image_layers(image)
#    display_image(image_rgb_layers,titles)


#df['weather']=
#get_certainty_description(df['split'].iloc[1], 0)


#pd.pivot_table(df, index=['weather_desc'])
#reg = re.compile("^weather\W[\d]+\W[\d]+.csv")
#file = "weather-51442-201606.csv"
#reg.match(file)


if __name__ == '__main__':
    main()
