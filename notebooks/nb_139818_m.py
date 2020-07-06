#!/usr/bin/env python
import os
my_dir_path = os.path.dirname(os.path.realpath(__file__))
log = "This is a log file of input/output vars of each cell.\n"


def print_info(x):
    import numpy as np
    res = ""
    if isinstance(x, list) or isinstance(x, np.ndarray):
        res = "shape" + str(np.shape(x)) + ";" + str(np.array(x).dtype)
    elif isinstance(x, dict):
        res = str(len(x)) + ";" + str(type(x))
    else:
        res = str(x) + ";" + str(type(x))
    return res.replace("\n", "")


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

log = log + "cell[1];OUT;X;" + print_info(X) + "\n"
log = log + "cell[1];OUT;iris;" + print_info(iris) + "\n"

# In[ ]:
log = log + "cell[2];IN;X;" + print_info(X) + "\n"

from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

# Once fitted, the pca model exposes the singular vectors in the components_ attribute:

log = log + "cell[2];OUT;pca;" + print_info(pca) + "\n"
log = log + "cell[2];OUT;X;" + print_info(X) + "\n"
log = log + "cell[2];OUT;PCA;" + print_info(PCA) + "\n"

# In[ ]:
log = log + "cell[3];IN;pca;" + print_info(pca) + "\n"

pca.components_

# Other attributes are available as well:

# In[ ]:
log = log + "cell[4];IN;pca;" + print_info(pca) + "\n"

pca.explained_variance_ratio_

# In[ ]:
log = log + "cell[5];IN;pca;" + print_info(pca) + "\n"

pca.explained_variance_ratio_.sum()

# Let us project the iris dataset along those first two dimensions:

# In[ ]:
log = log + "cell[6];IN;X;" + print_info(X) + "\n"
log = log + "cell[6];IN;pca;" + print_info(pca) + "\n"

X_pca = pca.transform(X)

# PCA `normalizes` and `whitens` the data, which means that the data
# is now centered on both components with unit variance:

log = log + "cell[6];OUT;X_pca;" + print_info(X_pca) + "\n"

# In[ ]:
log = log + "cell[7];IN;X_pca;" + print_info(X_pca) + "\n"

import numpy as np
np.round(X_pca.mean(axis=0), decimals=5)

log = log + "cell[7];OUT;np;" + print_info(np) + "\n"
log = log + "cell[7];OUT;X_pca;" + print_info(X_pca) + "\n"

# In[ ]:
log = log + "cell[8];IN;X_pca;" + print_info(X_pca) + "\n"
log = log + "cell[8];IN;np;" + print_info(np) + "\n"

np.round(X_pca.std(axis=0), decimals=5)

# Furthermore, the samples components do no longer carry any linear correlation:

log = log + "cell[8];OUT;X_pca;" + print_info(X_pca) + "\n"

# In[ ]:
log = log + "cell[9];IN;np;" + print_info(np) + "\n"
log = log + "cell[9];IN;X_pca;" + print_info(X_pca) + "\n"

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
        plt.scatter(data[target == i, 0],
                    data[target == i, 1],
                    c=c,
                    label=label)
    plt.legend()


# Now calling this function for our data, we see the plot:

log = log + "cell[11];OUT;plot_PCA_2D;" + print_info(plot_PCA_2D) + "\n"
log = log + "cell[11];OUT;plt;" + print_info(plt) + "\n"

# In[ ]:
log = log + "cell[12];IN;X_pca;" + print_info(X_pca) + "\n"
log = log + "cell[12];IN;plot_PCA_2D;" + print_info(plot_PCA_2D) + "\n"
log = log + "cell[12];IN;iris;" + print_info(iris) + "\n"

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

log = log + "cell[13];OUT;X;" + print_info(X) + "\n"
log = log + "cell[13];OUT;y;" + print_info(y) + "\n"

# In[19]:
log = log + "cell[14];IN;X;" + print_info(X) + "\n"
log = log + "cell[14];IN;PCA;" + print_info(PCA) + "\n"
log = log + "cell[14];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[14];IN;y;" + print_info(y) + "\n"

X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

# Manifold learning algorithms, however, available in the ``sklearn.manifold``
# submodule, are able to recover the underlying 2-dimensional manifold:

log = log + "cell[14];OUT;X;" + print_info(X) + "\n"

# In[ ]:
log = log + "cell[15];IN;X;" + print_info(X) + "\n"
log = log + "cell[15];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[15];IN;y;" + print_info(y) + "\n"

from sklearn.manifold import LocallyLinearEmbedding, Isomap
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified')
X_lle = lle.fit_transform(X)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y)

log = log + "cell[15];OUT;Isomap;" + print_info(Isomap) + "\n"
log = log + "cell[15];OUT;X;" + print_info(X) + "\n"

# In[ ]:
log = log + "cell[16];IN;Isomap;" + print_info(Isomap) + "\n"
log = log + "cell[16];IN;X;" + print_info(X) + "\n"
log = log + "cell[16];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[16];IN;y;" + print_info(y) + "\n"

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

f = open(os.path.join(my_dir_path, "nb_139818_m_log.txt"), "w")
f.write(log)
f.close()
