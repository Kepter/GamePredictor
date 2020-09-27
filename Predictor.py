import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

def preprocess(data):
    # Change Is_Home_or_Away to numeric
    data.loc[(data['Is_Home_or_Away'] == 'Home'), 'Is_Home_or_Away'] = 0
    data.loc[(data['Is_Home_or_Away'] == 'Away'), 'Is_Home_or_Away'] = 1

    # Change Is_Opponent_in_AP25_Preseason to numeric
    data.loc[(data['Is_Opponent_in_AP25_Preseason'] == 'Out'), 'Is_Opponent_in_AP25_Preseason'] = 0
    data.loc[(data['Is_Opponent_in_AP25_Preseason'] == 'In'), 'Is_Opponent_in_AP25_Preseason'] = 1

    # Change Media to numeric
    data.loc[(data['Media'] == '1-NBC'), 'Media'] = 0
    data.loc[(data['Media'] == '2-ESPN'), 'Media'] = 1
    data.loc[(data['Media'] == '3-FOX'), 'Media'] = 2
    data.loc[(data['Media'] == '4-ABC'), 'Media'] = 3
    data.loc[(data['Media'] == '5-CBS'), 'Media'] = 4

    # Extract month from date and transform to 0, 1, 2
    for i in range(len(data['Date'])):
        if data['Date'][i].startswith("9"):
            data['Date'][i] = 0
        elif data['Date'][i].startswith("10"):
            data['Date'][i] = 1
        elif data['Date'][i].startswith("11"):
            data['Date'][i] = 2
        
    # Change label to numeric
    data.loc[(data['Label'] == 'Lose'), 'Label'] = 0
    data.loc[(data['Label'] == 'Win'), 'Label'] = 1


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

preprocess(train)
preprocess(test)

# Drop irrelevant columns
train = train.drop(['ID', 'Opponent'], axis=1)
test = test.drop(['ID', 'Opponent'], axis=1)

# Split data
x_train = train.drop(['Label'], axis=1).astype(int)
y_train = train['Label'].astype(int)

x_test = test.drop(['Label'], axis=1).astype(int)
y_test = test['Label'].astype(int)

# Train and test Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)

naive_bayes_pred = nb.predict(x_test)

# Train and test KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)