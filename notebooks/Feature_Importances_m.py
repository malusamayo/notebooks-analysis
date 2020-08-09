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

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ## Titanic data + random columns

# In[5]:


titanic = pd.read_csv("titanic3.csv")

my_store_info((2, 1, "titanic"), titanic)

# In[6]:my_store_info((3, 0, "titanic"), titanic)



rng = np.random.RandomState(42)
titanic['random_cat'] = rng.randint(3, size=titanic.shape[0])
titanic['random_num'] = rng.randn(titanic.shape[0])

my_store_info((3, 1, "titanic"), titanic)

# In[7]:my_store_info((4, 0, "titanic"), titanic)



titanic.describe()


# In[8]:my_store_info((5, 0, "titanic"), titanic)



titanic.head(10)


# In[9]:my_store_info((6, 0, "titanic"), titanic)



categorical_columns = ['pclass', 'sex', 'embarked', 'random_cat']
numerical_columns = ['age', 'sibsp', 'parch', 'fare', 'random_num']

data = titanic[categorical_columns + numerical_columns]
labels = titanic['survived']

my_store_info((6, 1, "data"), data)
my_store_info((6, 1, "labels"), labels)
my_store_info((6, 1, "categorical_columns"), categorical_columns)
my_store_info((6, 1, "numerical_columns"), numerical_columns)

# In[10]:my_store_info((7, 0, "data"), data)
my_store_info((7, 0, "labels"), labels)



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    data, labels, stratify=labels, random_state=42)


# ## Building a Feature Engineering Pipeline
my_store_info((7, 1, "X_train"), X_train)
my_store_info((7, 1, "y_train"), y_train)
my_store_info((7, 1, "X_test"), X_test)
my_store_info((7, 1, "y_test"), y_test)

# In[11]:my_store_info((8, 0, "categorical_columns"), categorical_columns)
my_store_info((8, 0, "numerical_columns"), numerical_columns)



from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    # no scaler needed for random forests
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

rf = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', RandomForestClassifier(n_estimators=100,
                                          min_samples_leaf=1,
                                          random_state=42))
])

my_store_info((8, 1, "rf"), rf)

# In[12]:my_store_info((9, 0, "rf"), rf)
my_store_info((9, 0, "X_train"), X_train)
my_store_info((9, 0, "y_train"), y_train)



rf.fit(X_train, y_train)

my_store_info((9, 1, "X_train"), X_train)
my_store_info((9, 1, "y_train"), y_train)
my_store_info((9, 1, "rf"), rf)

# In[13]:my_store_info((10, 0, "X_train"), X_train)
my_store_info((10, 0, "y_train"), y_train)
my_store_info((10, 0, "rf"), rf)
my_store_info((10, 0, "X_test"), X_test)
my_store_info((10, 0, "y_test"), y_test)



print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))


# ## Tree-based Feature Importances

# In[14]:my_store_info((11, 0, "rf"), rf)
my_store_info((11, 0, "categorical_columns"), categorical_columns)
my_store_info((11, 0, "numerical_columns"), numerical_columns)



ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = []
for col, cats in zip(categorical_columns, ohe.categories_):
    for cat in cats:
        feature_names.append("{}_{}".format(col, cat))
feature_names = np.array(feature_names + numerical_columns)

tree_feature_importances = (
    rf.named_steps['classifier'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
_, ax = plt.subplots(figsize=(12, 8))
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances");


# ## Permutation-based Feature Importances (training set)

# In[24]:my_store_info((12, 0, "X_train"), X_train)
my_store_info((12, 0, "y_train"), y_train)
my_store_info((12, 0, "rf"), rf)
my_store_info((12, 0, "X_test"), X_test)



permute_importance = permutation_importance(rf, X_train, y_train, n_repeats=30,
                                            random_state=42).importances

permute_importance_mean = np.mean(permute_importance, axis=-1)
sorted_idx = permute_importance_mean.argsort()

_, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(permute_importance[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.vlines(0, 0, X_test.shape[1] + 1, linestyles='dashed', alpha=0.5)
ax.set_xlabel("baseline score - score on permutated variable")
ax.set_title("Permutation Importances (training set)");


# ## Permutation-based Feature Importances (test set)

# In[25]:my_store_info((13, 0, "rf"), rf)
my_store_info((13, 0, "X_test"), X_test)
my_store_info((13, 0, "y_test"), y_test)



permute_importance = permutation_importance(rf, X_test, y_test, n_repeats=30,
                                            random_state=42).importances
permute_importance_mean = np.mean(permute_importance, axis=-1)
sorted_idx = permute_importance_mean.argsort()

_, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(permute_importance[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.vlines(0, 0, X_test.shape[1] + 1, linestyles='dashed', alpha=0.5)
ax.set_xlabel("baseline score - score on permutated variable")
ax.set_title("Permutation Importances (test set)");


# ## Same Analysis With a Non-Overfitting Classifier

# In[26]:my_store_info((14, 0, "rf"), rf)
my_store_info((14, 0, "X_train"), X_train)
my_store_info((14, 0, "y_train"), y_train)
my_store_info((14, 0, "X_test"), X_test)
my_store_info((14, 0, "y_test"), y_test)



rf.named_steps['classifier'].set_params(min_samples_leaf=8)
rf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))

my_store_info((14, 1, "rf"), rf)
my_store_info((14, 1, "X_train"), X_train)
my_store_info((14, 1, "y_train"), y_train)

# In[27]:my_store_info((15, 0, "rf"), rf)
my_store_info((15, 0, "categorical_columns"), categorical_columns)
my_store_info((15, 0, "numerical_columns"), numerical_columns)



ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = []
for col, cats in zip(categorical_columns, ohe.categories_):
    for cat in cats:
        feature_names.append("{}_{}".format(col, cat))
feature_names = np.array(feature_names + numerical_columns)

tree_feature_importances = (
    rf.named_steps['classifier'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
_, ax = plt.subplots(figsize=(12, 8))
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances");


# In[28]:my_store_info((16, 0, "X_train"), X_train)
my_store_info((16, 0, "y_train"), y_train)
my_store_info((16, 0, "rf"), rf)
my_store_info((16, 0, "X_test"), X_test)



permute_importance = permutation_importance(rf, X_train, y_train, n_repeats=30,
                                            random_state=42).importances
permute_importance_mean = np.mean(permute_importance, axis=-1)
sorted_idx = permute_importance_mean.argsort()

_, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(permute_importance[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.vlines(0, 0, X_test.shape[1] + 1, linestyles='dashed', alpha=0.5)
ax.set_xlabel("baseline score - score on permutated variable")
ax.set_title("Permutation Importances (training set)");


# In[29]:my_store_info((17, 0, "rf"), rf)
my_store_info((17, 0, "X_test"), X_test)
my_store_info((17, 0, "y_test"), y_test)



permute_importance = permutation_importance(rf, X_test, y_test, n_repeats=30,
                                            random_state=0).importances
permute_importance_mean = np.mean(permute_importance, axis=-1)
sorted_idx = permute_importance_mean.argsort()

_, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(permute_importance[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.vlines(0, 0, X_test.shape[1] + 1, linestyles='dashed', alpha=0.5)
ax.set_xlabel("baseline score - score on permutated variable")
ax.set_title("Permutation Importances (test set)");


# In[ ]:




store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "Feature_Importances_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
