from sklearn.feature_extraction.text import CountVectorizer
import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import pickle
from sklearn.pipeline import Pipeline
from clean import clean

csv = pandas.read_csv("train.csv",header=None)

matrix = csv.as_matrix()
line1 = matrix[:,3]
line2 = matrix[:,4]

q1_values = [clean(str(m))+clean(str(n)) for m,n in zip(line1, line2)]

Y_train = np.array(map(str,matrix[:,5])).astype(np.int)
print Y_train.shape

'''
count_vect = CountVectorizer()
X_train1 = count_vect.fit_transform(q1_values)

tf_transformer = TfidfTransformer().fit(X_train1)
X_train_tf1 = tf_transformer.transform(X_train1)

print X_train_tf1.shape
'''

classifier = Pipeline([('vect', CountVectorizer()),('tfidf',TfidfTransformer()),('LogReg',LogisticRegression())])

clf = classifier.fit(q1_values, Y_train)

filename = 'quora.pickle'
file = open(filename, "wb")
pickle.dump(clf, file)

print "training done"
