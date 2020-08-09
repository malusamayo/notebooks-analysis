#!/usr/bin/env pythonimport os
import pickle
import copy
store_vars = []
my_labels = []
my_dir_path = os.path.dirname(os.path.realpath(__file__))
ignore_types = ["<class 'module'>"]
copy_types = [
    "<class 'folium.plugins.marker_cluster.MarkerCluster'>",
    "<class 'matplotlib.axes._subplots.AxesSubplot'>"
]
def my_store_info(info, var):
    if str(type(var)) in ignore_types:
        return
    my_labels.append(info)
    if str(type(var)) in copy_types:
        store_vars.append(copy.copy(var))
    else:
        store_vars.append(copy.deepcopy(var))

# coding: utf-8

# ### Scenario
# Alice is competing in Kaggle's Titanic competitions. Following the advice from an internet article, she tried heavy feature engineering and a simple model. However, she found that the model did not perform as well as what the original article claimed. She wondered what went wrong, but couldn't find the problem despite much effort. **Could you help her find problems in her notebook and improve the model's performance?**
# 
# ### Instruction
# + You are free to use the Internet to search for any problems you might encounter. However, please do not try to search the code directly.
# + You overall goal is to improve the model's performance, but you might prioritize finding the underlying problems in the notebook.
# 

# In[1170]:




import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
pd.options.display.max_columns = 100

from matplotlib import pyplot as plt
import numpy as np


# # II - Feature engineering
my_store_info((1, 1, "pd"), pd)

# In[1171]:


def status(feature):
    print('Processing', feature, ': ok')


# ###  Loading the data
# 
# One trick when starting a machine learning problem is to append the training set to the test set together.
# 
# We'll engineer new features using the train set to prevent information leakage. Then we'll add these variables to the test set.
# 
# Let's load the train and test sets and append them together.

# In[1172]:my_store_info((3, 0, "pd"), pd)



# reading train data
train = pd.read_csv('./data/train.csv')
train_len = len(train)

# reading test data
test = pd.read_csv('./data/test.csv')

# extracting and then removing the targets from the training data 
targets = train.Survived
train.drop(['Survived'], 1, inplace=True)


# merging train data and test data for future feature engineering
# we'll also remove the PassengerID since this is not an informative feature
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'PassengerId'], inplace=True, axis=1)


# ### Processing Title
my_store_info((3, 1, "combined"), combined)
my_store_info((3, 1, "train_len"), train_len)

# In[1173]:my_store_info((4, 0, "combined"), combined)



titles = set()
for name in combined['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())


# In[1174]:my_store_info((5, 0, "combined"), combined)



Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

# we extract the title from each name
combined['Title'] = combined['Name'].map(lambda name:name.split('.')[0].strip())

# a map of more aggregated title
# we map each title
combined['Title'] = combined.Title.map(Title_Dictionary)
status('Title')

my_store_info((5, 1, "combined"), combined)

# In[1175]:my_store_info((6, 0, "combined"), combined)



combined[combined['Title'].isnull()]
combined.at[combined['Title'].isnull(), "Title"] = "Royalty"


# ### Processing Ages
my_store_info((6, 1, "combined"), combined)

# In[1176]:my_store_info((7, 0, "combined"), combined)
my_store_info((7, 0, "train_len"), train_len)



grouped_train = combined.iloc[:train_len].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

my_store_info((7, 1, "grouped_median_train"), grouped_median_train)

# In[1177]:my_store_info((8, 0, "grouped_median_train"), grouped_median_train)
my_store_info((8, 0, "combined"), combined)



def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]

combined['Age'] = combined.apply(lambda row: fill_age(row), axis=1)
status('age')


# Let's now process the names.
my_store_info((8, 1, "combined"), combined)

# In[1178]:my_store_info((9, 0, "combined"), combined)
my_store_info((9, 0, "pd"), pd)



# we clean the Name variable
combined.drop('Name', axis=1, inplace=True)

# encoding in dummy variable
titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, titles_dummies], axis=1)

# removing the title variable
combined.drop('Title', axis=1, inplace=True)

status('names')


# ### Processing Fare
my_store_info((9, 1, "combined"), combined)

# In[1179]:my_store_info((10, 0, "combined"), combined)
my_store_info((10, 0, "train_len"), train_len)



# there's one missing fare value - replacing it with the mean.
combined.Fare.fillna(combined.iloc[:train_len].Fare.mean(), inplace=True)
status('fare')


# ### Processing Embarked

# In[1180]:my_store_info((11, 0, "combined"), combined)
my_store_info((11, 0, "pd"), pd)



# two missing embarked values - filling them with the most frequent one in the train  set(S)
combined.Embarked.fillna('S', inplace=True)
# dummy encoding 
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)
status('embarked')
combined.Cabin


# ### Processing Cabin
my_store_info((11, 1, "combined"), combined)

# In[1181]:my_store_info((12, 0, "combined"), combined)
my_store_info((12, 0, "pd"), pd)



# replacing missing cabins with U (for Uknown)
combined.Cabin.fillna('U', inplace=True)

# mapping each Cabin value with the cabin letter
combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

# dummy encoding ...
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
combined = pd.concat([combined, cabin_dummies], axis=1)

combined.drop('Cabin', axis=1, inplace=True)
status('cabin')


# This function replaces NaN values with U (for <i>Unknow</i>). It then maps each Cabin value to the first letter.
# Then it encodes the cabin values using dummy encoding again.

# Ok no missing values now.

# ### Processing Sex
my_store_info((12, 1, "combined"), combined)

# In[1182]:my_store_info((13, 0, "combined"), combined)



# mapping string values to numerical one 
combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
status('Sex')


# This function maps the string values male and female to 1 and 0 respectively. 

# ### Processing Pclass
my_store_info((13, 1, "combined"), combined)

# In[1183]:my_store_info((14, 0, "pd"), pd)
my_store_info((14, 0, "combined"), combined)



# encoding into 3 categories:
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

# adding dummy variable
combined = pd.concat([combined, pclass_dummies],axis=1)

# removing "Pclass"
combined.drop('Pclass',axis=1,inplace=True)

status('Pclass')


# This function encodes the values of Pclass (1,2,3) using a dummy encoding.

# ### Processing Ticket

# Let's first see how the different ticket prefixes we have in our dataset
my_store_info((14, 1, "combined"), combined)

# In[1184]:my_store_info((15, 0, "combined"), combined)
my_store_info((15, 0, "pd"), pd)



# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# Extracting dummy variables from tickets:

combined['Ticket'] = combined['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)

status('Ticket')


# ### Processing Family
my_store_info((15, 1, "combined"), combined)

# In[1185]:my_store_info((16, 0, "combined"), combined)



# introducing a new feature : the size of families (including the passenger)
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

# introducing other features based on the family size
combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

status('family')


# # III - Modeling
my_store_info((16, 1, "combined"), combined)

# In[1186]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# In[1187]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# Recovering the train set and the test set from the combined dataset is an easy task.

# In[1188]:my_store_info((19, 0, "pd"), pd)
my_store_info((19, 0, "combined"), combined)



targets = pd.read_csv('./data/train.csv', usecols=['Survived'])['Survived'].values
train = combined.iloc[:891]
test = combined.iloc[891:]


# ## Feature selection
my_store_info((19, 1, "train"), train)
my_store_info((19, 1, "targets"), targets)
my_store_info((19, 1, "test"), test)

# In[1189]:my_store_info((20, 0, "train"), train)
my_store_info((20, 0, "targets"), targets)



clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

my_store_info((20, 1, "train"), train)
my_store_info((20, 1, "clf"), clf)
my_store_info((20, 1, "targets"), targets)

# In[1190]:my_store_info((21, 0, "pd"), pd)
my_store_info((21, 0, "train"), train)
my_store_info((21, 0, "clf"), clf)



features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))


# ![energy](./images/article_1/8.png)

# In[1191]:my_store_info((22, 0, "clf"), clf)
my_store_info((22, 0, "train"), train)
my_store_info((22, 0, "test"), test)



model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
test_reduced = model.transform(test)


# ### Let's try a simple model
my_store_info((22, 1, "model"), model)
my_store_info((22, 1, "train_reduced"), train_reduced)

# In[1192]:my_store_info((23, 0, "model"), model)
my_store_info((23, 0, "train_reduced"), train_reduced)
my_store_info((23, 0, "targets"), targets)



model = RandomForestClassifier()
print('Cross-validation of : {0}'.format(model.__class__))
score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
print('CV score = {0}'.format(score))

store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "debug_example_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
