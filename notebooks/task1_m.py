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

# # Instructions
# + You are going to read the code, prepare to answer the following questions:
#     + What is the purpose of the 3rd cell and 4th cell?
#     + Where does the variable **X_train/X_test** come from? How?
# + Please reuse the code for new input "bank2.csv". Find what you need to change. 

# In[ ]:


import pandas as pd # Employ Pandas for data manipulation
import numpy as np

# 1. Import data
bank_df = pd.read_csv('bank1.csv', sep=";") # default is ",", which will fail

bank_df_raw = bank_df.copy() # back up the original dataset for multiple tests

my_store_info((1, 1, "bank_df"), bank_df)
my_store_info((1, 1, "bank_df_raw"), bank_df_raw)

# In[ ]:my_store_info((2, 0, "bank_df"), bank_df)



cat_cols = bank_df.select_dtypes(['object']).columns

my_store_info((2, 1, "cat_cols"), cat_cols)

# In[ ]:my_store_info((3, 0, "bank_df_raw"), bank_df_raw)
my_store_info((3, 0, "cat_cols"), cat_cols)



from sklearn import preprocessing
le = preprocessing.LabelEncoder()

bank_df_le = bank_df_raw.copy()

for col in cat_cols:    
    bank_df_le[col] = le.fit_transform(bank_df_le[col])

my_store_info((3, 1, "bank_df_raw"), bank_df_raw)
my_store_info((3, 1, "bank_df_le"), bank_df_le)

# In[ ]:my_store_info((4, 0, "bank_df_raw"), bank_df_raw)
my_store_info((4, 0, "bank_df_le"), bank_df_le)



month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
              'jul': 7, 'aug': 8, 'sep':9, 'oct': 10, 'nov': 11, 'dec': 12}

bank_df_fix = bank_df_raw.copy()

bank_df_le["month"] = bank_df_fix["month"].map(month_dict)
bank_df_le.loc[bank_df_le['pdays']==-1,'pdays'] = 999

my_store_info((4, 1, "bank_df_le"), bank_df_le)

# In[ ]:my_store_info((5, 0, "bank_df_le"), bank_df_le)



from sklearn.model_selection import train_test_split

X = bank_df_le.iloc[:,0:-1]
y = bank_df_le.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
# set random_state=1 so that the results will be reproducible every time the code was run

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

my_store_info((5, 1, "X_train"), X_train)
my_store_info((5, 1, "y_train"), y_train)
my_store_info((5, 1, "X_test"), X_test)
my_store_info((5, 1, "y_test"), y_test)

# In[ ]:my_store_info((6, 0, "X_train"), X_train)
my_store_info((6, 0, "y_train"), y_train)
my_store_info((6, 0, "X_test"), X_test)
my_store_info((6, 0, "y_test"), y_test)



from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print('LG Train Accuracy: %.3f' % pipe_lr.score(X_train, y_train))
print('LG Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

pipe_rf = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier(random_state=1))])
pipe_rf.fit(X_train, y_train)
print('RF Train Accuracy: %.3f' % pipe_rf.score(X_train, y_train))
print('RF Test Accuracy: %.3f' % pipe_rf.score(X_test, y_test))

my_store_info((6, 1, "pipe_rf"), pipe_rf)

# In[ ]:my_store_info((7, 0, "pipe_rf"), pipe_rf)
my_store_info((7, 0, "X_train"), X_train)
my_store_info((7, 0, "y_train"), y_train)



from sklearn.model_selection import GridSearchCV
param_grid_rf = [{'clf__n_estimators': [20,40,80], 'clf__criterion': ['gini', 'entropy'], 
                  'clf__max_features': [3,10], 'clf__max_depth': [3, None], 
                  'clf__min_samples_split':[3,10], 'clf__min_samples_leaf':[3, 10],
                  'clf__bootstrap': [True, False]}]
gs_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1)
gs_rf_fit = gs_rf.fit(X_train,y_train)

my_store_info((7, 1, "gs_rf_fit"), gs_rf_fit)

# In[ ]:my_store_info((8, 0, "gs_rf_fit"), gs_rf_fit)
my_store_info((8, 0, "X_train"), X_train)
my_store_info((8, 0, "y_train"), y_train)
my_store_info((8, 0, "X_test"), X_test)
my_store_info((8, 0, "y_test"), y_test)




print('Best score for RF: %.3f' % gs_rf_fit.best_score_)
print('Best param for RF: %s' % gs_rf_fit.best_params_)

gs_rf_best = gs_rf_fit.best_estimator_
print('RF Train accuracy: %.3f' % gs_rf_best.score(X_train, y_train))
print('RF Test accuracy: %.3f' % gs_rf_best.score(X_test, y_test))

my_store_info((8, 1, "gs_rf_best"), gs_rf_best)
my_store_info((8, 1, "X_train"), X_train)
my_store_info((8, 1, "y_train"), y_train)
my_store_info((8, 1, "X_test"), X_test)
my_store_info((8, 1, "y_test"), y_test)

# In[ ]:my_store_info((9, 0, "gs_rf_best"), gs_rf_best)
my_store_info((9, 0, "X_train"), X_train)
my_store_info((9, 0, "y_train"), y_train)



from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(estimator=gs_rf_best, X=X_train, 
                                                        y=y_train, train_sizes = np.linspace(0.1,1,10),
                                                        cv=10, n_jobs=-1)

# Mean and Std across K-Folds, which result in mean and std for each subset of X_train with different sample size
train_scores_mean = np.mean(train_scores, axis=1)
print('train_scores_mean: ', train_scores_mean)
train_scores_std = np.std(train_scores, axis=1)
print('train_scores_std: ', train_scores_std)
test_scores_mean = np.mean(test_scores, axis=1)
print('test_scores_mean: ', test_scores_mean)
test_scores_std = np.std(test_scores, axis=1)
print('test_scores_std: ', test_scores_std)

my_store_info((9, 1, "train_sizes"), train_sizes)
my_store_info((9, 1, "train_scores_mean"), train_scores_mean)
my_store_info((9, 1, "train_scores_std"), train_scores_std)
my_store_info((9, 1, "test_scores_mean"), test_scores_mean)
my_store_info((9, 1, "test_scores_std"), test_scores_std)
my_store_info((9, 1, "gs_rf_best"), gs_rf_best)
my_store_info((9, 1, "X_train"), X_train)
my_store_info((9, 1, "y_train"), y_train)

# In[ ]:my_store_info((10, 0, "train_sizes"), train_sizes)
my_store_info((10, 0, "train_scores_mean"), train_scores_mean)
my_store_info((10, 0, "train_scores_std"), train_scores_std)
my_store_info((10, 0, "test_scores_mean"), test_scores_mean)
my_store_info((10, 0, "test_scores_std"), test_scores_std)



import matplotlib.pyplot as plt 


plt.plot(train_sizes, train_scores_mean, color='blue', marker='o', markersize=5, label='training accuracy')

plt.fill_between(train_sizes, train_scores_mean+train_scores_std, 
                 train_scores_mean-train_scores_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_scores_mean, color='green', linestyle='--', marker='s', 
         markersize=5, label='validation accuracy')

plt.fill_between(train_sizes, test_scores_mean+test_scores_std, 
                 test_scores_mean-test_scores_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])

plt.show()


# In[ ]:my_store_info((11, 0, "gs_rf_best"), gs_rf_best)
my_store_info((11, 0, "X_train"), X_train)
my_store_info((11, 0, "y_train"), y_train)
my_store_info((11, 0, "X_test"), X_test)
my_store_info((11, 0, "y_test"), y_test)



from sklearn.metrics import confusion_matrix

clf_final = gs_rf_best
clf_final.fit(X_train, y_train)
y_pred = clf_final.predict(X_test)
print('Test accuracy: %.3f' % clf_final.score(X_test, y_test))

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

my_store_info((11, 1, "confmat"), confmat)

# In[ ]:my_store_info((12, 0, "confmat"), confmat)



import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.show()

store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "task1_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
