from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np

'''
Dataset from https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products?select=Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv

Creators of dataset: https://datafiniti.co/products/product-data/


Dataset contains data of amazon products updated between February 2019 and April 2019.
All fields are pre flattened'''



path = 'GradSchoolCode\Founations of Artifical Intelligence\Portfolio Project\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
df = pd.read_csv(path)

#drop columns that are not useful or redundant
df.drop(['reviews.username', 'reviews.didPurchase', 'reviews.id', 'reviews.doRecommend','sourceURLs', 'reviews.sourceURLs', 'imageURLs','reviews.numHelpful','asins','id','reviews.text','reviews.dateSeen','keys','manufacturerNumber','reviews.title','categories', 'brand'], axis = 1, inplace=True,errors = 'ignore')

#Drops items with less than 10 reviews
value_counts = df['name'].value_counts()
for value in range(len(value_counts)):
    if value_counts[value] < 10:
        index = df.loc[df.name == value_counts.keys()[value]]['name']
        df = df[df.name.isin(index) == False]

#converts dates to yyyymmdd
datelist = ['dateAdded', 'dateUpdated', 'reviews.date']
for date in datelist:
    df[date] = df[date].apply(lambda x: x[0:4] + x[5:7] + x[8:10])



#Seperate review ratings from training data
y = df['reviews.rating']
X = df.drop(['reviews.rating'], axis = 1)

#Make a dict mapping primaryCategories to numeric values

category_dict = dict()
count = 0
for cat in df['primaryCategories'].value_counts().keys():
    category_dict[cat] = count
    count += 1
X['primaryCategories'] = X['primaryCategories'].map(category_dict)
X['primaryCategories']

#Map names to numeric values
name_dict = dict()
count = 0
for name in df['name'].value_counts().keys():
    name_dict[name] = count
    count += 1
X['name'] = X['name'].map(name_dict)
X['name']



#Map manufacturer to numeric values
manufacturer_dict = dict()
count = 0
for man in df['manufacturer'].value_counts().keys():
    manufacturer_dict[man] = count
    count += 1

X['manufacturer'] = X['manufacturer'].map(manufacturer_dict)


#move the y value down one for better network training
y = y.map(lambda x: x-1)

#convert all dtypes in X to int from objects
X = X.astype(int)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)



#create the model
model = Sequential()
#hidden layers
model.add(Dense(10,activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10 ,activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))



model.add(Dense(units = 5, activation = 'sigmoid'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()



#train the model
#optimizer checked: Adagrad , sgd, RMSprop, Adadelta, Adam 
#RMSprop seems to work best, Adam second best, and Adadelta third, which makes sense because Adadelta is based off the same predecessor optimizor as RMSprop and Adam is similar to RMSprop
model.compile(loss = loss_fn, optimizer = 'RMSprop', metrics = 'accuracy')
model.fit(X_train, y_train, epochs = 5, batch_size = 20)


#check model evaluation
eval = model.evaluate(X_test, y_test)
print(eval)
print("The final accuracy for the network is {:.2f} %".format(eval[1] * 100))