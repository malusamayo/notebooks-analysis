#!/usr/bin/env pythonimport os
import pickle
import copy
store_vars = []
my_labels = []
my_dir_path = os.path.dirname(os.path.realpath(__file__))
def my_store_info(info, var):
    if str(type(var)) == "<class 'module'>":
        return
    my_labels.append(info)
    store_vars.append(copy.deepcopy(var))

# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[3]:


from io import StringIO
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz


def treeviz(tree): 
    dot_data = StringIO()  
    export_graphviz(tree, out_file=dot_data,  
                    feature_names=['petal (cm)', 'sepal (cm)'],  
                    class_names=iris.target_names,  
                    filled=True, rounded=True,  
                    special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())  


# In[4]:


def plot_model_decision(model, proba=False):
    plt.figure(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(0, 9, 100),
                         np.linspace(0, 9, 100))

    if proba:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.3)


    for i, label in enumerate(iris.target_names):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=label)

    plt.xlabel('Petal (cm)')
    plt.ylabel('Sepal (cm)')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.legend(loc='best');


# In[5]:


iris = load_iris()
iris.target_names

my_store_info((4, 1, "iris"), iris)

# In[6]:my_store_info((5, 0, "iris"), iris)



iris.feature_names


# In[7]:my_store_info((6, 0, "iris"), iris)



X = iris.data[:, [2, 0]]
y = iris.target

my_store_info((6, 1, "X"), X)
my_store_info((6, 1, "y"), y)

# In[8]:my_store_info((7, 0, "X"), X)
my_store_info((7, 0, "y"), y)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=50, random_state=42)

my_store_info((7, 1, "X_train"), X_train)
my_store_info((7, 1, "y_train"), y_train)
my_store_info((7, 1, "X_test"), X_test)
my_store_info((7, 1, "y_test"), y_test)

# In[9]:my_store_info((8, 0, "iris"), iris)
my_store_info((8, 0, "X"), X)
my_store_info((8, 0, "y"), y)



plt.figure(figsize=(8, 8))
for i, label in enumerate(iris.target_names):
    plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=label)

plt.xlabel('Petal (cm)')
plt.ylabel('Sepal (cm)')
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.legend(loc='best');


# In[10]:my_store_info((9, 0, "X_train"), X_train)
my_store_info((9, 0, "y_train"), y_train)



from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)
treeviz(tree)

my_store_info((9, 1, "tree"), tree)
my_store_info((9, 1, "X_train"), X_train)
my_store_info((9, 1, "y_train"), y_train)

# In[11]:my_store_info((10, 0, "tree"), tree)



plot_model_decision(tree)


# In[12]:my_store_info((11, 0, "X_train"), X_train)
my_store_info((11, 0, "y_train"), y_train)



from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression().fit(X_train[y_train != 0], y_train[y_train != 0])
plot_model_decision(lr_model, proba=True)

my_store_info((11, 1, "lr_model"), lr_model)

# In[13]:my_store_info((12, 0, "lr_model"), lr_model)



lr_model.coef_


# In[14]:my_store_info((13, 0, "lr_model"), lr_model)



lr_model.intercept_


# In[20]:my_store_info((14, 0, "X_train"), X_train)
my_store_info((14, 0, "y_train"), y_train)



from sklearn.linear_model import Perceptron

linear_model = Perceptron(max_iter=50)
linear_model.fit(X_train[y_train != 0], y_train[y_train != 0])

my_store_info((14, 1, "linear_model"), linear_model)

# In[21]:my_store_info((15, 0, "linear_model"), linear_model)
my_store_info((15, 0, "X_test"), X_test)
my_store_info((15, 0, "y_test"), y_test)



linear_model.score(X_test[y_test != 0], y_test[y_test != 0])

my_store_info((15, 1, "linear_model"), linear_model)

# In[22]:my_store_info((16, 0, "linear_model"), linear_model)



linear_model.coef_


# In[23]:my_store_info((17, 0, "linear_model"), linear_model)



linear_model.intercept_


# In[24]:my_store_info((18, 0, "linear_model"), linear_model)



plot_model_decision(linear_model)

store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "iris_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
