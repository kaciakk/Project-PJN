import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.plotting import plot_confusion_matrix
import xgboost as xgb
import string
import joblib
import re
import os
import shutil


# Rozpakowanie plików dancyh
shutil.unpack_archive("train.csv.zip", "train.csv")
shutil.unpack_archive("test.csv.zip", "test.csv")

# Wczytywanie danych
def read_data(filename, encoding="utf8"):
    df = pd.read_csv(filename)
    # Drop null values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("\nNumber of records:", len(df))
    return df

df = read_data("train.csv/train.csv")

# Wybór kolumn
df = df[["question1","question2","is_duplicate"]]
df.head() 

#Przetwarzanie wstępne dancych
def preprocess_data(text):
    # conver to string
    text = str(text)
    # lowercase
    text = text.lower()
    # remove contractions
    text = contractions.fix(text)
    # remove hashtags
    text = re.sub(r'#(\w+)','',text)
    # remove special characters
    text = re.sub(r'[^\w ]+','',text)
    # remove links if any
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    # remove non-ascii
    text = ''.join(word for word in text if ord(word) < 128)
    # remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))
    # remove digits
    text = re.sub(r'[\d]+','',text)
    # remove single letters
    text = ' '.join(word for word in text.split() if len(word) > 1)
    # remove multiple spaces
    text = ' '.join(text.split())

    return text

#Przetwarzanie wstępne na kopii
df_copy = df.copy() 
df_copy.loc[:,"question1"] = df_copy["question1"].apply(preprocess_data)
df_copy.loc[:,"question2"] = df_copy["question2"].apply(preprocess_data)
df_copy.head()

#Generowanie liczby wystąpień dla unikalnej wartości
df_copy['is_duplicate'].value_counts()

# Podział danych
df_train,df_dev = train_test_split(df_copy,
                                   test_size=0.3,
                                   stratify=df_copy['is_duplicate'],
                                   random_state=42)

print("Training data shape:",df_train.shape)
print("Dev data shape:",df_dev.shape)

#Wektoryzacja TF-IDF
tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(df_train['question1']+' '+df_train['question2'])
dev_tfidf = tfidf.transform(df_dev['question1']+' '+df_dev['question2'])

train_tfidf.shape,dev_tfidf.shape

#Trenowanie modelu
labels = df_train['is_duplicate']
model_xgb = xgb.XGBClassifier()
model_xgb.fit(train_tfidf,labels)

#Predykcja i ocena wyników 
predictions = model_xgb.predict(dev_tfidf)
predictions = list(predictions)

print('Accuracy score:',accuracy_score(df_dev['is_duplicate'],predictions))

# Confusion Matrix
cm = confusion_matrix(df_dev['is_duplicate'],predictions)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()