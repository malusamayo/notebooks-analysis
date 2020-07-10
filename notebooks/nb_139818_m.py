#!/usr/bin/env pythonimport os
import pickle
store_vars = []
my_labels = []
my_dir_path = os.path.dirname(os.path.realpath(__file__))

# coding: utf-8

# # Unsupervised Learning: Dimensionality Reduction and Visualization

# Unsupervised learning is interested in situations in which X is available, but not y: data without labels.
# 
# A typical use case is to find hiden structure in the data.
# 
# Previously we worked on visualizing the iris data by plotting
# pairs of dimensions by trial and error, until we arrived at
# the best pair of dimensions for our dataset.  Here we will
# use an unsupervised *dimensionality reduction* algorithm
# to accomplish this more automatically.

# By the end of this section you will
# 
# - Know how to instantiate and train an unsupervised dimensionality reduction algorithm:
#   Principal Component Analysis (PCA)
# - Know how to use PCA to visualize high-dimensional data

# ## Dimensionality Reduction: PCA

# Dimensionality reduction is the task of deriving a set of new
# artificial features that is smaller than the original feature
# set while retaining most of the variance of the original data.
# Here we'll use a common but powerful dimensionality reduction
# technique called Principal Component Analysis (PCA).
# We'll perform PCA on the iris dataset that we saw before:

# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target


# PCA is performed using linear combinations of the original features
# using a truncated Singular Value Decomposition of the matrix X so
# as to project the data onto a base of the top singular vectors.
# If the number of retained components is 2 or 3, PCA can be used
# to visualize the dataset.
my_labels.append((1, 1, "X"))
store_vars.append(X)
my_labels.append((1, 1, "iris"))
store_vars.append(iris)

# In[ ]:my_labels.append((2, 0, "X"))
store_vars.append(X)



from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(X)


# Once fitted, the pca model exposes the singular vectors in the components_ attribute:
my_labels.append((2, 1, "pca"))
store_vars.append(pca)
my_labels.append((2, 1, "X"))
store_vars.append(X)
my_labels.append((2, 1, "PCA"))
store_vars.append(PCA)

# In[ ]:my_labels.append((3, 0, "pca"))
store_vars.append(pca)



pca.components_


# Other attributes are available as well:

# In[ ]:my_labels.append((4, 0, "pca"))
store_vars.append(pca)



pca.explained_variance_ratio_


# In[ ]:my_labels.append((5, 0, "pca"))
store_vars.append(pca)



pca.explained_variance_ratio_.sum()


# Let us project the iris dataset along those first two dimensions:

# In[ ]:my_labels.append((6, 0, "X"))
store_vars.append(X)
my_labels.append((6, 0, "pca"))
store_vars.append(pca)



X_pca = pca.transform(X)


# PCA `normalizes` and `whitens` the data, which means that the data
# is now centered on both components with unit variance:
my_labels.append((6, 1, "X_pca"))
store_vars.append(X_pca)

# In[ ]:my_labels.append((7, 0, "X_pca"))
store_vars.append(X_pca)



import numpy as np
np.round(X_pca.mean(axis=0), decimals=5)

my_labels.append((7, 1, "X_pca"))
store_vars.append(X_pca)

# In[ ]:my_labels.append((8, 0, "X_pca"))
store_vars.append(X_pca)



np.round(X_pca.std(axis=0), decimals=5)


# Furthermore, the samples components do no longer carry any linear correlation:
my_labels.append((8, 1, "X_pca"))
store_vars.append(X_pca)

# In[ ]:my_labels.append((9, 0, "X_pca"))
store_vars.append(X_pca)



np.corrcoef(X_pca.T)


# We can visualize the projection using pylab, but first
# let's make sure our ipython notebook is in pylab inline mode

# In[ ]:





# Now we can visualize the results using the following utility function:

# In[ ]:


import matplotlib.pyplot as plt
from itertools import cycle

def plot_PCA_2D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    plt.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label)
    plt.legend()


# Now calling this function for our data, we see the plot:

# In[ ]:my_labels.append((12, 0, "X_pca"))
store_vars.append(X_pca)
my_labels.append((12, 0, "iris"))
store_vars.append(iris)



plot_PCA_2D(X_pca, iris.target, iris.target_names)


# Note that this projection was determined *without* any information about the
# labels (represented by the colors): this is the sense in which the learning
# is **unsupervised**.  Nevertheless, we see that the projection gives us insight
# into the distribution of the different flowers in parameter space: notably,
# *iris setosa* is much more distinct than the other two species.

# Note also that the default implementation of PCA computes the
# singular value decomposition (SVD) of the full
# data matrix, which is not scalable when both ``n_samples`` and
# ``n_features`` are big (more that a few thousands).
# If you are interested in a number of components that is much
# smaller than both ``n_samples`` and ``n_features``, consider using
# `sklearn.decomposition.RandomizedPCA` instead.

# Other dimensionality reduction techniques which are useful to know about:
# 
# - [sklearn.decomposition.PCA](http://scikit-learn.org/0.13/modules/generated/sklearn.decomposition.PCA.html): 
#    Principal Component Analysis
# - [sklearn.decomposition.RandomizedPCA](http://scikit-learn.org/0.13/modules/generated/sklearn.decomposition.RandomizedPCA.html):
#    fast non-exact PCA implementation based on a randomized algorithm
# - [sklearn.decomposition.SparsePCA](http://scikit-learn.org/0.13/modules/generated/sklearn.decomposition.SparsePCA.html):
#    PCA variant including L1 penalty for sparsity
# - [sklearn.decomposition.FastICA](http://scikit-learn.org/0.13/modules/generated/sklearn.decomposition.FastICA.html):
#    Independent Component Analysis
# - [sklearn.decomposition.NMF](http://scikit-learn.org/0.13/modules/generated/sklearn.decomposition.NMF.html):
#    non-negative matrix factorization
# - [sklearn.manifold.LocallyLinearEmbedding](http://scikit-learn.org/0.13/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html):
#    nonlinear manifold learning technique based on local neighborhood geometry
# - [sklearn.manifold.IsoMap](http://scikit-learn.org/0.13/modules/generated/sklearn.manifold.Isomap.html):
#    nonlinear manifold learning technique based on a sparse graph algorithm

# ## Manifold Learning

# One weakness of PCA is that it cannot detect non-linear features.  A set
# of algorithms known as *Manifold Learning* have been developed to address
# this deficiency.  A canonical dataset used in Manifold learning is the
# *S-curve*, which we briefly saw in an earlier section:

# In[ ]:


from sklearn.datasets import make_s_curve
X, y = make_s_curve(n_samples=1000)

from mpl_toolkits.mplot3d import Axes3D
# ax = plt.axes(projection='3d')

# ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)
# ax.view_init(10, -60)


# This is a 2-dimensional dataset embedded in three dimensions, but it is embedded
# in such a way that PCA cannot discover the underlying data orientation:
my_labels.append((13, 1, "X"))
store_vars.append(X)
my_labels.append((13, 1, "y"))
store_vars.append(y)

# In[19]:my_labels.append((14, 0, "X"))
store_vars.append(X)
my_labels.append((14, 0, "PCA"))
store_vars.append(PCA)
my_labels.append((14, 0, "y"))
store_vars.append(y)



X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)


# Manifold learning algorithms, however, available in the ``sklearn.manifold``
# submodule, are able to recover the underlying 2-dimensional manifold:
my_labels.append((14, 1, "X"))
store_vars.append(X)

# In[ ]:my_labels.append((15, 0, "X"))
store_vars.append(X)
my_labels.append((15, 0, "y"))
store_vars.append(y)



from sklearn.manifold import LocallyLinearEmbedding, Isomap
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified')
X_lle = lle.fit_transform(X)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y)

my_labels.append((15, 1, "Isomap"))
store_vars.append(Isomap)
my_labels.append((15, 1, "X"))
store_vars.append(X)

# In[ ]:my_labels.append((16, 0, "Isomap"))
store_vars.append(Isomap)
my_labels.append((16, 0, "X"))
store_vars.append(X)
my_labels.append((16, 0, "y"))
store_vars.append(y)



iso = Isomap(n_neighbors=15, n_components=2)
X_iso = iso.fit_transform(X)
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y)


# ## Exercise: Dimension reduction of digits

# Apply PCA LocallyLinearEmbedding, and Isomap to project the data to two dimensions.
# Which visualization technique separates the classes most cleanly?

# In[ ]:


from sklearn.datasets import load_digits
digits = load_digits()
# ...


# In[ ]:





# ### Solution:

# In[ ]:


# %load solutions/08A_digits_projection.py

store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "nb_139818_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
