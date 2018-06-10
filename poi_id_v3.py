# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:44:50 2018

@author: James
"""

import sys
import pickle
import pprint
from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from time import time
import pandas as pd
import numpy as np

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

#First glance at the data
print "Begin Exploratory Data Analysis (EDA)\n"
print "Show Ken Lay's data in the dictionary:"
pprint.pprint(data_dict["LAY KENNETH L"], width =1) 
print "\nRemoved email address from dictionary, unnecessary and has @ character\n"
for key, value in data_dict.items():
    del value['email_address']
print "Ken Lay minus email address:"
pprint.pprint(data_dict["LAY KENNETH L"], width =1) 

#Converting the dataset from a python dictionary to a pandas dataframe for EDA
df = pd.DataFrame.from_dict(data_dict, orient='index')
print "\nTake a look at the data frame.\n"
print df.head()
print

#High level look at the data frame
print "\nNumber of individuals in dataset: ", len(df) 

#Looking for NaN values
#replace NaN string value with numpy NaN
df.replace('NaN', np.nan, inplace = True)
#https://stackoverflow.com/questions/38922952/how-to-replace-all-the-nan-strings-with-empty-string-in-my-dataframe
print "\nTop 10 individuals with NaN feature values:\n "
print df.isnull().sum(axis=1).sort_values(ascending=False)[:10]
#https://stackoverflow.com/questions/30059260/python-pandas-counting-the-number-of-missing-nan-in-each-row
print "\nEugene Lockhart's feature values:\n"
print df.loc["LOCKHART EUGENE E"]
print "\nTop 10 features with NaN values:\n"
print df.isnull().sum(axis=0).sort_values(ascending=False)[:10]

#Persons of Interest
poi_counts = df['poi'].value_counts().tolist()
print "\npoi_counts:"
print df['poi'].value_counts().sort_values(ascending=False)
#http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.2/cookbook/Chapter%202%20-%20Selecting%20data%20%26%20finding%20the%20most%20common%20complaint%20type.ipynb
df_poi = df[(df.poi==True)]
#https://pythonspot.com/pandas-filter/
print "\nList of persons of interest:"
print list(df_poi.index)

#Remove Outliers
print '\nObserved Outlier # 1, "Total"...this is not a person, it is the sum \
line of the data which lists, for example, a salary of ', \
data_dict["TOTAL"]['salary']
print
print 'Observed Outlier # 2, "THE TRAVEL AGENCY IN THE PARK"...this is not a \
person. '
print
print 'Observed Outlier # 3, "LOCKHART EUGENE E"...other than poi, all \
features are NaN. '
print
del data_dict["TOTAL"] #remove Outlier # 1
del data_dict["THE TRAVEL AGENCY IN THE PARK"] #remove Outlier # 2
del data_dict["LOCKHART EUGENE E"] #remove Outlier # 3

print "Number of individuals in dataset after removing the outliers: ", \
len(data_dict)
print

"""create new feature to show what percentage of each individual's emails 
involved POI's"""
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
print
print features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
""" From featureFormat in Tools folder:
        convert dictionary to numpy array of features
        with the following parameters:
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """
labels, features = targetFeatureSplit(data)
""" From featureFormat in Tools folder:
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

"""Classifiers tested in Pipeline, commented out all but top classifier"""
#classifier = RandomForestClassifier()
#classifier = DecisionTreeClassifier()
classifier = GaussianNB()
#classifier = LinearSVC()
#classifier = KNeighborsClassifier()
#classifier = SVC()

scaler = MinMaxScaler()
select = SelectKBest()
n_features = list(range(1, len(features_list)))

"""Parameter grid for GridSearch...commented out params for all but top\
 classifier"""
param_grid = [
    {
        'feature_selection__k': n_features,
    },
     { #for_tree classifiers
#        'classifier__criterion': ('gini','entropy'),
#        'classifier__splitter':('best','random'), #not used by RandomForest
#        'classifier__min_samples_split':[2, 10, 20],
#        'classifier__max_depth':[None,10,15,20,25,30],
#        'classifier__max_leaf_nodes':[None,5,10,30]
#      #for_LinearSVC   
#         'classifier__penalty': ('l1','l2'),
#         'classifier__loss': ('hinge','squared_hinge')
       #for_KNeighbors
#           'classifier__n_neighbors': [3,4,5],
#           'classifier__weights': ('uniform','distance')
       #for_SVC
#           'classifier__kernel': ('rbf','linear')
        }
]

pipe = Pipeline([
        ('scaler', scaler),
        ('feature_selection', select),
        ('classifier', classifier)])

cv = StratifiedShuffleSplit(labels,random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1')

start = time()
grid.fit(features,labels)

print
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid.cv_results_['params'])))
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
print

#Classifier Results
print """RandomForest test classifier scores:
    Accuracy:  0.85467
    Precision: 0.38971
    Recall:    0.15900 
    F1:        0.22585
    using SelectKBest k=10
    46.35 seconds for 164 candidate parameter settings"""
print
print """DecisionTreeClassifier test classifier scores:
    Accuracy:  0.82873
    Precision: 0.30202
    Recall:    0.21700 
    F1:        0.25255
    using SelectKBest k=10
    17.07 seconds for 308 candidate parameter settings"""
print
print """GaussianNB test classifier scores:
    Accuracy:  0.84487
    Precision: 0.39632
    Recall:    0.31250 
    F1:        0.34945
    using SelectKBest k=11
    0.92 seconds for 21 candidate parameter settings"""
print
print """LinearSVC test classifier scores:
    Accuracy:  0.86320
    Precision: 0.45324
    Recall:    0.12600 
    F1:        0.19718
    using SelectKBest k=6
    1.20 seconds for 22 candidate parameter settings"""
print
print """SVC test classifier scores:
    Accuracy:  0.86633
    Precision: 0.48344
    Recall:    0.03650 
    F1:        0.06788
    using SelectKBest k=1
    1.19 seconds for 22 candidate parameter settings"""

print
print "GridSearchCV best parameters:", grid.best_params_
print
print "GaussianNB classifier has the best scores."

#Get Selected Features
print
selected_indices = grid.best_estimator_.named_steps["feature_selection"].\
get_support(indices=True)
print "Selected feature indices: ", selected_indices
selected_indices_list = selected_indices.tolist()
print
selected_features = [features_list[i+1] for i in selected_indices]
#https://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index
print "Selected Features: ",selected_features
print

#Name selected estimator clf for easy export below
clf = grid.best_estimator_

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "Validate grid-selected estimator using cross-validation:"

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

#Dump classifier, dataset, and features_list so anyone can check the results
dump_classifier_and_data(clf, my_dataset, features_list)

print
print "Test classifier results:"
test_classifier(clf, my_dataset, features_list)
