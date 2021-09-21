import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt





df = pd.read_csv(r'C:\Users\SAI\OneDrive\Desktop\Venu_News\News Train.csv')




df



df.head()





df = df.drop(['ArticleId'],axis=1)





df.head()




from sklearn.model_selection import train_test_split




df['Category'].unique()




data = df['Text']




target=df['Category']





from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()


df.insert(2, "ENCODED_CATEGORY", labelencoder.fit_transform(df['Category']), True)



df.head()



X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3,random_state=42)




len(X_train)




X_train.shape





X_test.shape




len(y_train)




from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer =TfidfVectorizer(stop_words='english',ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)





X_train.shape





X_test.shape




y_train.shape


 


y_train





clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
acc=accuracy_score(y_test,pred)
print(acc)
print(confusion_matrix(y_test,pred))





new4=['2019 World Cup | Rain takes over after Williamson and Taylor lead New Zealands fightback']
new4=vectorizer.transform(new4)
p=clf.predict(new4.todense())
print(p)





new4=['Sensex drops over 250 points; TCS down 2%']
new4=vectorizer.transform(new4)
p=clf.predict(new4.todense())
print(p)



new4=['Dr. Reddy’s launches drug for cold in the U.S']
new4=vectorizer.transform(new4)
p=clf.predict(new4.todense())
print(p)





new=['Bharti Airtel users consume about 11GB data per month; overtake Reliance JioNEWS Bharti Airtel users consume about 11GB data per month; overtake Reliance Jio']
new=vectorizer.transform(new)
p=clf.predict(new.todense())
print(p)





new1=['5G download speed nearly 3 times faster than 4G in US: Report']
new1=vectorizer.transform(new1)
p=clf.predict(new1.todense())
print(p)





clf=RandomForestClassifier(n_estimators=10,random_state=42,n_jobs=-1)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
acc=accuracy_score(y_test,pred)
print(acc)
print(confusion_matrix(y_test,pred))





clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
acc=accuracy_score(y_test,pred)
print(acc)
print(confusion_matrix(y_test,pred))

