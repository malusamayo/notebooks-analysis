#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ## Titanic data + random columns

# In[5]:


titanic = pd.read_csv("titanic3.csv")


# In[6]:


rng = np.random.RandomState(42)
titanic['random_cat'] = rng.randint(3, size=titanic.shape[0])
titanic['random_num'] = rng.randn(titanic.shape[0])


# In[7]:


titanic.describe()


# In[8]:


titanic.head(10)


# In[9]:


categorical_columns = ['pclass', 'sex', 'embarked', 'random_cat']
numerical_columns = ['age', 'sibsp', 'parch', 'fare', 'random_num']

data = titanic[categorical_columns + numerical_columns]
labels = titanic['survived']


# In[10]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    data, labels, stratify=labels, random_state=42)


# ## Building a Feature Engineering Pipeline

# In[11]:


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


# In[12]:


rf.fit(X_train, y_train)


# In[13]:


print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))


# ## Tree-based Feature Importances

# In[14]:


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

# In[24]:


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

# In[25]:


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

# In[26]:


rf.named_steps['classifier'].set_params(min_samples_leaf=8)
rf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))


# In[27]:


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


# In[28]:


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


# In[29]:


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




