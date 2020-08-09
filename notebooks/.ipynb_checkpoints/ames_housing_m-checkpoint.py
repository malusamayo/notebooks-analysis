#!/usr/bin/env python
import os
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

# # House price prediction
#
#
# ## Loading the Ames Housing Price dataset

# In[2]:

import pandas as pd

# In[3]:

AMES_HOUSING_CSV = "file2ed11cebe25.csv"
df = pd.read_csv(AMES_HOUSING_CSV)
df

# ## Some basic data visualization

my_store_info((2, 1, "df"), df)

# In[4]:
my_store_info((3, 0, "df"), df)

import folium
from folium.plugins import MarkerCluster

center = df[["Latitude", "Longitude"]].mean().values.tolist()
ames_map = folium.Map(location=center, zoom_start=13)
c = MarkerCluster(options={"maxClusterRadius": 40}).add_to(ames_map)


def make_tooltip(record):
    tooltip = f'<div>Sale_Price: ${record["Sale_Price"] / 1e3:.1f}k</div>'
    tooltip += f'<div>Gr_Liv_Area: {record["Gr_Liv_Area"]:.1f}</div>'
    tooltip += f'<div>Year_Built: {record["Year_Built"]:d}</div>'
    return tooltip


for i, record in df.iterrows():
    marker = folium.Marker((record["Latitude"], record["Longitude"]),
                           tooltip=make_tooltip(record))
    marker.add_to(c)

ames_map

my_store_info((3, 1, "c"), c)
my_store_info((3, 1, "i"), i)

# In[5]:
my_store_info((4, 0, "df"), df)

df.plot(x="Gr_Liv_Area",
        y="Sale_Price",
        kind="scatter",
        figsize=(12, 6),
        alpha=0.1)

# ## A baseline univariate linear model

# In[6]:
my_store_info((5, 0, "df"), df)

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Sale_Price"])
y = df["Sale_Price"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=500,
                                                    random_state=0)

my_store_info((5, 1, "X_train"), X_train)
my_store_info((5, 1, "X_test"), X_test)
my_store_info((5, 1, "y_train"), y_train)
my_store_info((5, 1, "y_test"), y_test)
my_store_info((5, 1, "X"), X)

# In[7]:
my_store_info((6, 0, "X_train"), X_train)
my_store_info((6, 0, "X_test"), X_test)
my_store_info((6, 0, "y_train"), y_train)
my_store_info((6, 0, "y_test"), y_test)

from sklearn.linear_model import LinearRegression

x_train = X_train["Gr_Liv_Area"].values.reshape(-1, 1)
x_test = X_test["Gr_Liv_Area"].values.reshape(-1, 1)

lr = LinearRegression().fit(x_train, y_train)
r2_score = lr.score(x_test, y_test)

print(f"R2 score (test): {r2_score:.3f}")

my_store_info((6, 1, "lr"), lr)
my_store_info((6, 1, "y_train"), y_train)

# In[8]:
my_store_info((7, 0, "lr"), lr)
my_store_info((7, 0, "X_test"), X_test)

y_pred = lr.predict(X_test["Gr_Liv_Area"].values.reshape(-1, 1))

my_store_info((7, 1, "y_pred"), y_pred)

# In[9]:
my_store_info((8, 0, "y_pred"), y_pred)
my_store_info((8, 0, "y_test"), y_test)

import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_pred, y_test, alpha=0.3)
    ax.plot([1e4, 5e5], [1e4, 5e5], linestyle="--")
    ax.set_xlim(1e4, 5e5)
    ax.set_ylim(1e4, 5e5)
    ax.set_xlabel("Predicted price ($)")
    ax.set_ylabel("Actual price ($)")


plot_predictions(y_test, y_pred)

my_store_info((8, 1, "y_test"), y_test)
my_store_info((8, 1, "y_pred"), y_pred)

# In[10]:
my_store_info((9, 0, "y_test"), y_test)
my_store_info((9, 0, "y_pred"), y_pred)

import numpy as np


def mean_absolute_percent_error(y_true, y_pred):
    diffs = np.abs(y_true - y_pred)
    scales = np.abs(y_true) + np.finfo(np.float64).eps
    return np.mean(diffs / scales) * 100


mape = mean_absolute_percent_error(y_test, y_pred)

print(f"MAPE (test): {mape:.3f}")

# ## Fitting a non linear, multi-variate model

# In[11]:
my_store_info((10, 0, "X_train"), X_train)

X_train.head()

my_store_info((10, 1, "X_train"), X_train)

# In[12]:
my_store_info((11, 0, "df"), df)


def caterogical_columns(df):
    return df.columns[df.dtypes == object]


len(caterogical_columns(df))

# In[13]:
my_store_info((12, 0, "df"), df)


def numeric_columns(df):
    return df.columns[df.dtypes != object]


len(numeric_columns(df))

# In[14]:
my_store_info((13, 0, "df"), df)
my_store_info((13, 0, "c"), c)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

categories = [df[c].unique() for c in caterogical_columns(df)]
ord_encoder = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer([
    ("categorical", ord_encoder, caterogical_columns),
    ("numeric", "passthrough", numeric_columns),
])

hgb = HistGradientBoostingRegressor(
    max_leaf_nodes=3,
    learning_rate=0.5,
    early_stopping=True,
    n_iter_no_change=10,
    max_bins=5,
    max_iter=1000,
    random_state=0,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", hgb),
])

my_store_info((13, 1, "model"), model)

# In[15]:
my_store_info((14, 0, "model"), model)
my_store_info((14, 0, "X_train"), X_train)
my_store_info((14, 0, "y_train"), y_train)

_ = model.fit(X_train, y_train)

my_store_info((14, 1, "model"), model)
my_store_info((14, 1, "X_train"), X_train)
my_store_info((14, 1, "y_train"), y_train)

# In[16]:
my_store_info((15, 0, "model"), model)

model[-1].n_iter_

# In[17]:
my_store_info((16, 0, "X_train"), X_train)
my_store_info((16, 0, "y_train"), y_train)
my_store_info((16, 0, "model"), model)

r2_score_train = model.score(X_train, y_train)
print(f"r2 score (train): {r2_score_train:.3f}")

# In[18]:
my_store_info((17, 0, "model"), model)
my_store_info((17, 0, "X_test"), X_test)
my_store_info((17, 0, "y_test"), y_test)

test_r2_score = model.score(X_test, y_test)
print(f"r2 score (test): {test_r2_score:.3f}")

# In[19]:
my_store_info((18, 0, "model"), model)
my_store_info((18, 0, "X_test"), X_test)
my_store_info((18, 0, "y_test"), y_test)
my_store_info((18, 0, "y_pred"), y_pred)

y_pred = model.predict(X_test)
mape = mean_absolute_percent_error(y_test, y_pred)
print(f"MAPE: {mape:.1f}%")

# In[20]:
my_store_info((19, 0, "model"), model)
my_store_info((19, 0, "X_test"), X_test)
my_store_info((19, 0, "y_test"), y_test)
my_store_info((19, 0, "y_pred"), y_pred)

y_pred = model.predict(X_test)
plot_predictions(y_test, y_pred)

# ## Selecting the most important variables

my_store_info((19, 1, "y_test"), y_test)
my_store_info((19, 1, "y_pred"), y_pred)

# In[21]:
my_store_info((20, 0, "model"), model)
my_store_info((20, 0, "X_test"), X_test)
my_store_info((20, 0, "y_test"), y_test)

from sklearn.inspection import permutation_importance

pi = permutation_importance(model,
                            X_test,
                            y_test,
                            n_repeats=5,
                            random_state=42,
                            n_jobs=2)

my_store_info((20, 1, "pi"), pi)

# In[22]:
my_store_info((21, 0, "pi"), pi)
my_store_info((21, 0, "i"), i)
my_store_info((21, 0, "df"), df)

sorted_idx = pi.importances_mean.argsort()
most_important_idx = [
    i for i in sorted_idx
    if pi.importances_mean[i] - 4 * pi.importances_std[i] > 0
]
most_important_names = df.columns[most_important_idx]
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(pi.importances[most_important_idx].T,
           vert=False,
           labels=most_important_names)
ax.set_title("Permutation Importances (test set)")

# ## Retraining a simpler model on the most important features only

my_store_info((21, 1, "most_important_names"), most_important_names)

# In[23]:
my_store_info((22, 0, "most_important_names"), most_important_names)

feature_subset = most_important_names.tolist()
if "Latitude" not in feature_subset:
    feature_subset += ["Latitude"]
if "Longitude" not in feature_subset:
    feature_subset += ["Longitude"]

my_store_info((22, 1, "feature_subset"), feature_subset)

# In[24]:
my_store_info((23, 0, "df"), df)
my_store_info((23, 0, "feature_subset"), feature_subset)

len(numeric_columns(df[feature_subset]))

# In[25]:
my_store_info((24, 0, "df"), df)
my_store_info((24, 0, "feature_subset"), feature_subset)

len(caterogical_columns(df[feature_subset]))

# In[26]:
my_store_info((25, 0, "X_train"), X_train)
my_store_info((25, 0, "feature_subset"), feature_subset)

X_train[feature_subset]

# In[27]:
my_store_info((26, 0, "df"), df)
my_store_info((26, 0, "c"), c)
my_store_info((26, 0, "feature_subset"), feature_subset)
my_store_info((26, 0, "X_train"), X_train)
my_store_info((26, 0, "y_train"), y_train)

categories = [df[c].unique() for c in caterogical_columns(df[feature_subset])]
ord_encoder = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer([
    ("categorical", ord_encoder, caterogical_columns),
    ("numeric", "passthrough", numeric_columns),
])
hgb = HistGradientBoostingRegressor(
    max_leaf_nodes=16,
    learning_rate=0.1,
    min_samples_leaf=5,
    early_stopping=True,
    n_iter_no_change=5,
    max_iter=1000,
    random_state=0,
)
reduced_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", hgb),
])

_ = reduced_model.fit(X_train[feature_subset], y_train)

my_store_info((26, 1, "reduced_model"), reduced_model)
my_store_info((26, 1, "y_train"), y_train)

# In[28]:
my_store_info((27, 0, "reduced_model"), reduced_model)
my_store_info((27, 0, "y_train"), y_train)
my_store_info((27, 0, "X_train"), X_train)
my_store_info((27, 0, "feature_subset"), feature_subset)

_ = reduced_model.fit(X_train[feature_subset], y_train)

my_store_info((27, 1, "reduced_model"), reduced_model)
my_store_info((27, 1, "y_train"), y_train)

# In[29]:
my_store_info((28, 0, "reduced_model"), reduced_model)

reduced_model[-1].n_iter_

# In[30]:
my_store_info((29, 0, "y_train"), y_train)
my_store_info((29, 0, "reduced_model"), reduced_model)
my_store_info((29, 0, "X_train"), X_train)
my_store_info((29, 0, "feature_subset"), feature_subset)

r2_score_train = reduced_model.score(X_train[feature_subset], y_train)
print(f"r2 score (train): {r2_score_train:.3f}")

# In[31]:
my_store_info((30, 0, "reduced_model"), reduced_model)
my_store_info((30, 0, "X_test"), X_test)
my_store_info((30, 0, "y_test"), y_test)
my_store_info((30, 0, "feature_subset"), feature_subset)

test_r2_score = reduced_model.score(X_test[feature_subset], y_test)
print(f"r2 score (test): {test_r2_score:.3f}")

# In[32]:
my_store_info((31, 0, "reduced_model"), reduced_model)
my_store_info((31, 0, "X_test"), X_test)
my_store_info((31, 0, "feature_subset"), feature_subset)
my_store_info((31, 0, "y_test"), y_test)
my_store_info((31, 0, "y_pred"), y_pred)

y_pred = reduced_model.predict(X_test[feature_subset])
mape = mean_absolute_percent_error(y_test, y_pred)
print(f"MAPE: {mape:.1f}%")

my_store_info((31, 1, "y_pred"), y_pred)

# In[33]:
my_store_info((32, 0, "y_test"), y_test)
my_store_info((32, 0, "y_pred"), y_pred)

plot_predictions(y_test, y_pred)

# ## Model inspection

my_store_info((32, 1, "y_test"), y_test)

# In[34]:
my_store_info((33, 0, "reduced_model"), reduced_model)
my_store_info((33, 0, "X_test"), X_test)
my_store_info((33, 0, "y_test"), y_test)
my_store_info((33, 0, "feature_subset"), feature_subset)

from sklearn.inspection import permutation_importance

pi = permutation_importance(reduced_model,
                            X_test[feature_subset],
                            y_test,
                            n_repeats=10,
                            random_state=42,
                            n_jobs=2)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(feature_subset)[sorted_idx]
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(pi.importances[sorted_idx].T, vert=False, labels=sorted_names)
ax.set_title("Permutation Importances (test set)")

# In[37]:

# %pip install -q git+https://github.com/slundberg/shap

# In[36]:
my_store_info((35, 0, "reduced_model"), reduced_model)
my_store_info((35, 0, "X_test"), X_test)
my_store_info((35, 0, "feature_subset"), feature_subset)

from sklearn.inspection import plot_partial_dependence

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model,
                        X_test[feature_subset], ["Year_Built"],
                        grid_resolution=20,
                        ax=ax)

my_store_info((35, 1, "reduced_model"), reduced_model)
my_store_info((35, 1, "ax"), ax)

# In[37]:
my_store_info((36, 0, "reduced_model"), reduced_model)
my_store_info((36, 0, "ax"), ax)
my_store_info((36, 0, "X_test"), X_test)
my_store_info((36, 0, "feature_subset"), feature_subset)

fig, ax = plt.subplots(figsize=(10, 5))
plot_partial_dependence(reduced_model,
                        X_test[feature_subset], ["Gr_Liv_Area"],
                        grid_resolution=20,
                        ax=ax)

my_store_info((36, 1, "reduced_model"), reduced_model)
my_store_info((36, 1, "ax"), ax)

# In[38]:
my_store_info((37, 0, "reduced_model"), reduced_model)
my_store_info((37, 0, "ax"), ax)
my_store_info((37, 0, "X"), X)
my_store_info((37, 0, "feature_subset"), feature_subset)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model,
                        X[feature_subset], [["Gr_Liv_Area", "Year_Built"]],
                        grid_resolution=20,
                        contour_kw={"alpha": 0.8},
                        ax=ax)

my_store_info((37, 1, "reduced_model"), reduced_model)
my_store_info((37, 1, "ax"), ax)

# In[39]:
my_store_info((38, 0, "reduced_model"), reduced_model)
my_store_info((38, 0, "ax"), ax)
my_store_info((38, 0, "X"), X)
my_store_info((38, 0, "feature_subset"), feature_subset)

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model,
                        X[feature_subset], [["Longitude", "Latitude"]],
                        percentiles=(0., 1.),
                        grid_resolution=20,
                        contour_kw={"alpha": 0.8},
                        ax=ax)
ax = fig.gca()
ax.set_xlim(X["Longitude"].min(), X["Longitude"].max())
ax.set_ylim(X["Latitude"].min(), X["Latitude"].max())
ax.set_aspect("equal")
ax.scatter(X["Longitude"], X["Latitude"])

# ## Model selection: hyperparameter tuning

# In[40]:

HistGradientBoostingRegressor()

# In[41]:
my_store_info((40, 0, "df"), df)
my_store_info((40, 0, "c"), c)

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

categories = [df[c].unique() for c in caterogical_columns(df)]
ord_encoder = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer([
    ("categorical", ord_encoder, caterogical_columns),
    ("numeric", "passthrough", numeric_columns),
])
hgb = HistGradientBoostingRegressor(
    early_stopping=True,
    n_iter_no_change=10,
    max_iter=1000,
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", hgb),
])

params = {
    "regressor__learning_rate": np.logspace(-3, 0, 10),
    "regressor__max_leaf_nodes": [2, 3, 4, 5, 6, 8, 16, 32, 64],
    "regressor__max_bins": [3, 5, 10, 30, 50, 100, 255],
    "regressor__min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}
search = RandomizedSearchCV(model,
                            params,
                            n_iter=200,
                            cv=3,
                            n_jobs=4,
                            verbose=1)

my_store_info((40, 1, "model"), model)

# In[44]:

# _ = search.fit(X_train, y_train)

# In[45]:

# cv_results = pd.DataFrame(search.cv_results_)
# cv_results = cv_results.sort_values("mean_test_score", ascending=False)
# cv_results.to_json("ames_gbrt_search_results.json")

# In[46]:

cv_results = pd.read_json("ames_gbrt_search_results.json")

my_store_info((43, 1, "cv_results"), cv_results)

# In[47]:


def rename_param(column_name):
    if "__" in column_name:
        return column_name.rsplit("__", 1)[1]
    return column_name


# In[48]:
my_store_info((45, 0, "cv_results"), cv_results)

cv_results.rename(rename_param, axis=1).head(5)

# ### Interactions between hyperparameters and generalization

# In[50]:
my_store_info((46, 0, "cv_results"), cv_results)

import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.rename(rename_param, axis=1).apply({
        "learning_rate":
        np.log10,
        "max_leaf_nodes":
        np.log2,
        "max_bins":
        np.log2,
        "mean_test_score":
        lambda x: x,
    }),
    color="mean_test_score",
    color_continuous_scale=px.colors.diverging.Portland,
)
fig.show()

# Let's zoom on the top performing models by using the `query` methods of the dataframe. Note that the axis have a narrower range now:

my_store_info((46, 1, "px"), px)
my_store_info((46, 1, "rename_param"), rename_param)
my_store_info((46, 1, "cv_results"), cv_results)

# In[51]:
my_store_info((47, 0, "px"), px)
my_store_info((47, 0, "rename_param"), rename_param)
my_store_info((47, 0, "cv_results"), cv_results)

fig = px.parallel_coordinates(
    cv_results.rename(rename_param, axis=1).apply({
        "learning_rate":
        np.log10,
        "max_leaf_nodes":
        np.log2,
        "max_bins":
        np.log2,
        "mean_test_score":
        lambda x: x,
    }).query("mean_test_score > 0.88"),
    color="mean_test_score",
    color_continuous_scale=px.colors.diverging.Portland,
)
fig.show()

# Let's check that the inner CV scores still approximately reflect the true generatlization score measured on held out data even when we select the best model from hundred of possible candidates via random search:

# In[52]:
my_store_info((48, 0, "cv_results"), cv_results)

best_search_result = cv_results.nlargest(n=1,
                                         columns=["mean_test_score"]).iloc[0]
print(
    f'R2 score of best candidate (inner CV): {best_search_result["mean_test_score"]:.3f}'
    f' (+/-{3 * best_search_result["std_test_score"]:.3f})')

my_store_info((48, 1, "best_search_result"), best_search_result)

# In[53]:
my_store_info((49, 0, "model"), model)
my_store_info((49, 0, "best_search_result"), best_search_result)
my_store_info((49, 0, "X_train"), X_train)
my_store_info((49, 0, "y_train"), y_train)
my_store_info((49, 0, "X_test"), X_test)
my_store_info((49, 0, "y_test"), y_test)

model.set_params(**best_search_result["params"])
model.fit(X_train, y_train)
held_out_score = model.score(X_test, y_test)
print(f'R2 score of best candidate on held-out data: {held_out_score:.3f}')

# ### Exploration of the accuracy / prediction latency trade-off

# In[54]:
my_store_info((50, 0, "cv_results"), cv_results)

cv_results["safe_test_score"] = cv_results["mean_test_score"] - cv_results[
    "std_test_score"]

my_store_info((50, 1, "cv_results"), cv_results)

# In[55]:
my_store_info((51, 0, "c"), c)
my_store_info((51, 0, "cv_results"), cv_results)

import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()

param_names = [c for c in cv_results.columns if c.startswith("param_")]
fig = px.scatter(cv_results,
                 x="mean_score_time",
                 y="safe_test_score",
                 hover_data=param_names)

fig.show()

# In[ ]:

store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "ames_housing_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
