# Copula-Based Outlier Detection COPOD
# A copula describes the dependence structure between random variables
# Copulas are functions that enable us to separate marginal distributions 
# from the dependency structure of a given multivariate distribution.

#Outlier score
# A higher anomaly score indicates that there is a low probability of the data instance 
# because it lies on the tail of the data distribution.


#data generation

from pyod.utils.data import generate_data
import numpy as np
X_train, y_train, X_test, y_test = \
        generate_data(n_train=200,
                      n_test=100,
                      n_features=5,
                      contamination=0.1,
                      random_state=3) 
X_train = X_train * np.random.uniform(0, 1, size=X_train.shape)
X_test = X_test * np.random.uniform(0,1, size=X_test.shape)
from pyod.models.copod import COPOD
clf_name = 'COPOD'
clf = COPOD()
clf.fit(X_train)
test_scores = clf.decision_function(X_test)

#Training and Detection

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
roc = round(roc_auc_score(y_test, test_scores), ndigits=4)  
prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

#Results

print(f'{clf_name} Receiver Operating Characteristic Score:{roc}, precision @ rank n:{prn}')