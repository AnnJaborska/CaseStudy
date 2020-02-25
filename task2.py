import json
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

# list = []
# for line in open('data.json', 'r'):
#     list.append(json.loads(line))
#
# headline  = []

# for i in range(len(list)):
#
#     headline.append(dict.get('Age')[i])
#
# print(headline)

df=pd.read_json("data.json", lines=True)
print(df)

print(df.is_sarcastic.value_counts()    )
len = df.is_sarcastic.count()

corpus = []
for i in range(0, len):
    review = re.sub('!', ' exclamation', df['headline'][i])
    review = review.replace('?', ' inquiry')
    matches = re.findall(r'\'(.+?)\'', review)
    if matches:
        review += ' quotation'
    review = re.sub('[^a-z]', ' ', review)
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)
y = df.iloc[:, 1]
# print(y)

from sklearn.feature_extraction.text import CountVectorizer

features_n = range(100, 3000, 100)
scores = []
for i in features_n:
    cv = CountVectorizer(max_features=i)
    X = cv.fit_transform(corpus).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    NB = GaussianNB()
    NB.fit(X_train, y_train)

    y_pred = NB.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    error_rate = (FP + FN) / (TP + TN + FP + FN)  # calculating the error rate based on confusion matrix results
    scores.append(error_rate)
    print(error_rate)
    print(classification_report(y_test, y_pred))

opt_n = features_n[scores.index(min(scores))]
print("The optimal number of max vectors is %d" % opt_n + " with an error rate of %.3f" % min(scores))
plt.plot(features_n, scores)
plt.xlabel('Number of Max Vectors')
plt.ylabel('Error Rate')
plt.show()
