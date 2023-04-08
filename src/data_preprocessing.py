import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def preprocess(df):
    x = df["Text"]
    y = df["Language"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    data_list = []

    for text in x:
        text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
        
        text = text.lower()
        data_list.append(text)

    cv = CountVectorizer()
    X = cv.fit_transform(data_list).toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    return x_train, x_test, y_train, y_test, le, cv

if __name__ == '__main__':
    pass