import pandas as pd
df=pd.read_csv('Restaurant_Reviews.tsv',sep='\t')

#to see first five entries we code -
df.head()
#output.
Review	Liked
0	Wow... Loved this place.	1
1	Crust is not good.	0
2	Not tasty and the texture was just nasty.	0
3	Stopped by during the late May bank holiday of...	1
4	The selection on the menu was great and so wer...	1

#to see last five entries we code-
df.tail()
#output
Review	Liked
995	I think food should have flavor and texture an...	0
996	Appetite instantly gone.	0
997	Overall I was not impressed and would not go b...	0
998	The whole experience was underwhelming, and I ...	0
999	Then, as if I hadn't wasted enough of my life ...	0

#to see the no of rows and columns 
df.shape

#to see the strucute and roles
df.info()

#to see the no of unique entrie and total count we code -
df.describe(include="object").T

df.describe().T

#to see the no of liked and disliked comment we code -
df['Liked'].value_counts()

#to see the lenght of the comments we code -
df['length']=df['Review'].apply(len)
df.head()

#to set the lenght of the comment and then look for the result we code -
df[df['length']==50]['Review'].iloc[0]

#importing libraies 
import nltk
nltk.download('stopwords')

import string
from nltk.corpus import stopwords

#to remove the stop words we code -
stopwords.words('english')

#to remove the puncutaions we code -
[punc for punc in string.punctuation]

#applying the code to remove it -

def text_process(msg):
    nopunc = [char for char in msg if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ''.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])
df.head()


#to compare the result and original side by side we code-
df['tokenized_Review']=df['Review'].apply(text_process)
df.head()

#creating word cloud -
#positive comments
from wordcloud import WordCloud
import matplotlib.pyplot as plt
word_cloud = df.loc[df['Liked']==1,:]
text = ' '.join([text for text in word_cloud['Review']])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#negative review
from wordcloud import WordCloud
import matplotlib.pyplot as plt
word_cloud = df.loc[df['Liked']==0,:]
text = ' '.join([text for text in word_cloud['Review']])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#importing the necessary libraries -
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

#case 1
from numpy import vectorize
vectorizer = CountVectorizer(max_df = 0.9, min_df = 10)
X = vectorizer.fit_transform(df['Review']).toarray()


#
Xfrom sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(df['Review'], df['Liked'], random_state= 107, test_size=0.2)
X_train.head()


#
train_vectorized = vectorizer.transform(X_train)
test_vectorized = vectorizer.transform(X_test)

X_train_array = train_vectorized.toarray()
X_test_array = test_vectorized.toarray()


#
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_array, y_train)

y_train_preds_nb = nb.predict(X_train_array)
y_test_preds_nb = nb.predict(X_test_array)
y_test_preds_nb

y_test


#
pd.DataFrame({"actual_y_value":y_test, "predicted_y_value":y_test_preds_nb})


#importing the necessary libraries -
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report

#TO check the accuracy
def print_metrics(actual, predicted):
  print('accuracy_score is {}'.format(accuracy_score(actual, predicted)))
  print('precision_score is {}'.format(precision_score(actual, predicted)))

  print('recall_score is {}'.format(recall_score(actual, predicted)))
  print('f1_score is {}'.format(f1_score(actual, predicted)))
  print('confusion_matirx is {}'.format(confusion_matrix(actual, predicted)))
  print('classification_report is {}'.format(classification_report(actual, predicted)))

print_metrics(y_train, y_train_preds_nb)


#
from sklearn.naive_bayes import MultinomialNB

mnv=MultinomialNB()
mnv.fit(X_train_array ,y_train)

y_train_preds_mnv = mnv.predict(X_train_array)
y_test_preds_mnv = mnv.predict(X_test_array)

y_test_preds_mnv

print_metrics(y_train, y_train_preds_mnv)
print_metrics(y_test, y_test_preds_mnv)

#to check the best accuracy of the result obtained we code - 
import numpy as np

best_accuracy =0.0
alpha_val=0

for i in np.arange(0.01,1.1,0.1):
  temp_cls=MultinomialNB(alpha=i)
  temp_cls.fit(X_train_array, y_train)
  y_test_preds_h_nbayes = temp_cls.predict(X_test_array)
  score=accuracy_score(y_test, y_test_preds_h_nbayes)
  print("accuracy score for alpha-{} is :{}%".format(round(i,1),round(score*100,2)))
  if score>best_accuracy:
    best_accuracy = score
    alpha_val =i
  print("....................................")
  print("the best accuracy is {}% with alpha value as {}".format(round(best_accuracy*100,2), round(alpha_val,1)))


Case 2 - 

#case2
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,
                     stop_words='english',
                     ngram_range = (1,1),
                     tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['Review'])
print(text_counts.shape)

#case 2
count_df = pd.DataFrame(text_counts.toarray(),columns=cv.get_feature_names_out())
count_df.head()

#case 2
count_df = pd.DataFrame(text_counts.toarray(),columns=cv.get_feature_names_out())

# case 2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts,
                                                    df['Liked'],
                                                    test_size=0.3,
                                                    random_state=1)

#case 2
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)
z = predicted
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, z))
