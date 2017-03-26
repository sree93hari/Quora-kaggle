import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas
import numpy as np

algo = pickle.load(open("quora.pickle","rb"))
'''
csv = pandas.read_csv("test.csv",header=None)

matrix = csv.as_matrix()

line1 = matrix[:,1]
line2 = matrix[:,2]
q1_values = [str(m)+str(n) for m,n in zip(line1, line2)]

ids = np.array(matrix[:,0])
vals =  algo.predict_proba(q1_values)
i = 0
res = open("results.csv","w")
res.write("test_id,is_duplicate\n")
for val in vals:
	res.write(str(ids[i]))
	res.write(",")
	res.write(str(val[1]))
	res.write("\n")
	i = i + 1 

res.close()
'''

X = ["How does the Surface Pro himself 4 compare with iPad Pro?Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?"]
print algo.predict_proba(X)
