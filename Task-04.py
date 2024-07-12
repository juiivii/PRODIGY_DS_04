import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#importing dataset 
train_data=pd.read_csv('twitter_training.csv',names=['TID','Topic','Impact','Comments'])
valid_data=pd.read_csv('twitter_validation.csv',names=['TID','Topic','Impact','Comments'])

#concatination of two datasets into one
new_dataset= pd.concat([train_data,valid_data],ignore_index=False)
print(new_dataset.head())
print(new_dataset['Topic'].value_counts())

#data cleaning
#to check if there are any na values
print(new_dataset.isna().sum())
new_dataset.dropna(inplace=True)
new_dataset.drop_duplicates(inplace=True)

plt.figure(figsize=(11, 10))
crosstab = pd.crosstab(index=new_dataset['Topic'], columns=new_dataset['Impact'])
sns.heatmap(crosstab, cmap = 'jet')
plt.show()

#Building model
X = new_dataset['Comments'] 
y = new_dataset['Impact'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

y_pred = rf_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

