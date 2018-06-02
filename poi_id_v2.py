# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:44:50 2018

@author: James
"""

import sys
import pickle
import pprint
from collections import OrderedDict
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages',
                 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']
# did not include email_address, unnecessary and had "@" character


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Take a look at the data set."
print
print "Number of individuals in dataset: ", len(data_dict)
print
print "Single example of dataset, Ken Lay: "
print
pprint.pprint(data_dict["LAY KENNETH L"], width =1)    
print

#find number of poi's in data set
print 'Persons of Interest (POI):'
count_poi = 0

for key, value in data_dict.items():
    if value['poi']:
        count_poi = count_poi + 1
        print key

print "number of persons of interest (poi): ", count_poi
print

### Task 2: Remove outliers

print 'Observed Outlier # 1, "Total"...this is not a person, it is the sum \
line of the data which lists, for example, a salary of ', \
data_dict["TOTAL"]['salary']
print
print 'Observed Outlier # 2, "THE TRAVEL AGENCY IN THE PARK"...this is not a \
person. '
print
del data_dict["TOTAL"] #remove Outlier # 1
del data_dict["THE TRAVEL AGENCY IN THE PARK"] #remove Outlier # 2

print "Number of individuals in dataset after removing the outliers: ", \
len(data_dict)
print

### Task 3: Create new feature(s)

#create new feature to show what percentage of each individual's emails 
#involved POI's
for key, value in data_dict.items():
    if value['from_messages'] != 'NaN' and value['to_messages'] != 'NaN':
        value['total_emails'] = value['from_messages'] + value['to_messages']
        value['total_poi_emails'] = value['from_poi_to_this_person']\
                                + value['from_this_person_to_poi']
        value['pct_poi_emails'] = float(value['total_poi_emails']) / \
        float(value['total_emails'])
    else:
        value['pct_poi_emails'] = 0
print "Show newly created feature for Ken Lay that gives the percentage of \
POI emails, 'pct_poi_emails', including both from and to:"
print

pprint.pprint(data_dict["LAY KENNETH L"], width =1) 
print

#sort dict descending by new feature
data_dict_descending = OrderedDict(sorted(data_dict.items(), 
                                          key=lambda kv: 
                                              kv[1]['pct_poi_emails'], 
                                              reverse = True))

print "Five highest pct_poi_emails values: "
print

for key, value in data_dict_descending.items()[:5]:
    print key, "percentage of poi emails = ", value['pct_poi_emails']

print
print "None of these five are in the POI list, but will test further."
print

### add new feature to features list
print 'Revised features list:'
features_list.append('pct_poi_emails')

print features_list
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
clf = DecisionTreeClassifier()
scaler = MinMaxScaler()

select = SelectKBest(k=10)

steps = [('scaler', scaler),
        ('feature_selection', select),
        ('classifier', clf)]

pipeline = Pipeline(steps)

pipeline.fit(features_train, labels_train)

y_prediction = pipeline.predict(features_test)
report = classification_report(labels_test, y_prediction )
print(report)

#parameters = dict(feature_selection__k =[1, 5])

parameters = {'classifier__criterion':('gini', 'entropy'), 'classifier__splitter':('best', 'random')}

gs = GridSearchCV(pipeline, param_grid=parameters)

gs.fit(features_train, labels_train)
y_predictions = gs.predict(features_test)
reports = classification_report(labels_test, y_predictions )
print reports
print gs

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
clf = gs

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from time import time
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy_metric = accuracy_score(pred,labels_test)
print "sklearn accuracy score = ",accuracy_metric

from sklearn.metrics import precision_score
precision = precision_score(labels_test, pred)
print "precision = ", precision

from sklearn.metrics import recall_score
recall = recall_score(labels_test, pred)
print "recall= ", recall

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print
print "Test classifier results:"
test_classifier(clf, my_dataset, features_list)