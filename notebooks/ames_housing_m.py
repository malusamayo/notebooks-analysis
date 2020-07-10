#!/usr/bin/env pythonimport os
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

# # House price prediction
# 
# 
# ## Loading the Ames Housing Price dataset

# In[2]:


import pandas as pd

log = log + "cell[1];OUT;pd;" + print_info(pd) + "\n"

# In[3]:log = log + "cell[2];IN;pd;" + print_info(pd) + "\n"



AMES_HOUSING_CSV = "https://www.openml.org/data/get_csv/20649135/file2ed11cebe25.arff"
df = pd.read_csv(AMES_HOUSING_CSV)
df


# ## Some basic data visualization
log = log + "cell[2];OUT;df;" + print_info(df) + "\n"

# In[4]:log = log + "cell[3];IN;df;" + print_info(df) + "\n"



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

log = log + "cell[3];OUT;c;" + print_info(c) + "\n"
log = log + "cell[3];OUT;i;" + print_info(i) + "\n"

# In[5]:log = log + "cell[4];IN;df;" + print_info(df) + "\n"



df.plot(x="Gr_Liv_Area", y="Sale_Price", kind="scatter",
        figsize=(12, 6), alpha=0.1);


# ## A baseline univariate linear model

# In[6]:log = log + "cell[5];IN;df;" + print_info(df) + "\n"



from sklearn.model_selection import train_test_split


X = df.drop(columns=["Sale_Price"])
y = df["Sale_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=500, random_state=0)

log = log + "cell[5];OUT;X_train;" + print_info(X_train) + "\n"
log = log + "cell[5];OUT;X_test;" + print_info(X_test) + "\n"
log = log + "cell[5];OUT;y_train;" + print_info(y_train) + "\n"
log = log + "cell[5];OUT;y_test;" + print_info(y_test) + "\n"
log = log + "cell[5];OUT;X;" + print_info(X) + "\n"

# In[7]:log = log + "cell[6];IN;X_train;" + print_info(X_train) + "\n"
log = log + "cell[6];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[6];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[6];IN;y_test;" + print_info(y_test) + "\n"



from sklearn.linear_model import LinearRegression

x_train = X_train["Gr_Liv_Area"].values.reshape(-1, 1)
x_test = X_test["Gr_Liv_Area"].values.reshape(-1, 1)

lr = LinearRegression().fit(x_train, y_train)
r2_score = lr.score(x_test, y_test)

print(f"R2 score (test): {r2_score:.3f}")

log = log + "cell[6];OUT;lr;" + print_info(lr) + "\n"
log = log + "cell[6];OUT;y_train;" + print_info(y_train) + "\n"

# In[8]:log = log + "cell[7];IN;lr;" + print_info(lr) + "\n"
log = log + "cell[7];IN;X_test;" + print_info(X_test) + "\n"



y_pred = lr.predict(X_test["Gr_Liv_Area"].values.reshape(-1, 1))

log = log + "cell[7];OUT;y_pred;" + print_info(y_pred) + "\n"

# In[9]:log = log + "cell[8];IN;y_pred;" + print_info(y_pred) + "\n"
log = log + "cell[8];IN;y_test;" + print_info(y_test) + "\n"



import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_pred, y_test, alpha=0.3)
    ax.plot([1e4, 5e5], [1e4, 5e5], linestyle="--")
    ax.set_xlim(1e4, 5e5); ax.set_ylim(1e4, 5e5)
    ax.set_xlabel("Predicted price ($)")
    ax.set_ylabel("Actual price ($)")


plot_predictions(y_test, y_pred)

log = log + "cell[8];OUT;y_pred;" + print_info(y_pred) + "\n"
log = log + "cell[8];OUT;y_test;" + print_info(y_test) + "\n"
log = log + "cell[8];OUT;plot_predictions;" + print_info(plot_predictions) + "\n"
log = log + "cell[8];OUT;plt;" + print_info(plt) + "\n"

# In[10]:log = log + "cell[9];IN;y_pred;" + print_info(y_pred) + "\n"



import numpy as np


def mean_absolute_percent_error(y_true, y_pred):
    diffs = np.abs(y_true - y_pred)
    scales = np.abs(y_true) + np.finfo(np.float64).eps
    return np.mean(diffs / scales) * 100


mape = mean_absolute_percent_error(y_test, y_pred)

print(f"MAPE (test): {mape:.3f}")


# ## Fitting a non linear, multi-variate model
log = log + "cell[9];OUT;mean_absolute_percent_error;" + print_info(mean_absolute_percent_error) + "\n"
log = log + "cell[9];OUT;np;" + print_info(np) + "\n"

# In[11]:log = log + "cell[10];IN;X_train;" + print_info(X_train) + "\n"



X_train.head()

log = log + "cell[10];OUT;X_train;" + print_info(X_train) + "\n"

# In[12]:log = log + "cell[11];IN;df;" + print_info(df) + "\n"



def caterogical_columns(df):
    return df.columns[df.dtypes == object]

len(caterogical_columns(df))

log = log + "cell[11];OUT;caterogical_columns;" + print_info(caterogical_columns) + "\n"

# In[13]:log = log + "cell[12];IN;df;" + print_info(df) + "\n"



def numeric_columns(df):
    return df.columns[df.dtypes != object]


len(numeric_columns(df))

log = log + "cell[12];OUT;numeric_columns;" + print_info(numeric_columns) + "\n"

# In[14]:log = log + "cell[13];IN;df;" + print_info(df) + "\n"
log = log + "cell[13];IN;c;" + print_info(c) + "\n"
log = log + "cell[13];IN;caterogical_columns;" + print_info(caterogical_columns) + "\n"
log = log + "cell[13];IN;numeric_columns;" + print_info(numeric_columns) + "\n"



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


categories = [df[c].unique() for c in caterogical_columns(df)]
ord_encoder =  OrdinalEncoder(categories=categories)

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

log = log + "cell[13];OUT;model;" + print_info(model) + "\n"
log = log + "cell[13];OUT;OrdinalEncoder;" + print_info(OrdinalEncoder) + "\n"
log = log + "cell[13];OUT;ColumnTransformer;" + print_info(ColumnTransformer) + "\n"
log = log + "cell[13];OUT;HistGradientBoostingRegressor;" + print_info(HistGradientBoostingRegressor) + "\n"
log = log + "cell[13];OUT;Pipeline;" + print_info(Pipeline) + "\n"

# In[15]:log = log + "cell[14];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[14];IN;model;" + print_info(model) + "\n"
log = log + "cell[14];IN;X_train;" + print_info(X_train) + "\n"




_ = model.fit(X_train, y_train)

log = log + "cell[14];OUT;model;" + print_info(model) + "\n"
log = log + "cell[14];OUT;y_train;" + print_info(y_train) + "\n"

# In[16]:log = log + "cell[15];IN;model;" + print_info(model) + "\n"



model[-1].n_iter_


# In[17]:log = log + "cell[16];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[16];IN;model;" + print_info(model) + "\n"



r2_score_train = model.score(X_train, y_train)
print(f"r2 score (train): {r2_score_train:.3f}")

log = log + "cell[16];OUT;model;" + print_info(model) + "\n"
log = log + "cell[16];OUT;X_train;" + print_info(X_train) + "\n"
log = log + "cell[16];OUT;y_train;" + print_info(y_train) + "\n"

# In[18]:log = log + "cell[17];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[17];IN;model;" + print_info(model) + "\n"
log = log + "cell[17];IN;y_test;" + print_info(y_test) + "\n"



test_r2_score = model.score(X_test, y_test)
print(f"r2 score (test): {test_r2_score:.3f}")

log = log + "cell[17];OUT;X_test;" + print_info(X_test) + "\n"
log = log + "cell[17];OUT;y_test;" + print_info(y_test) + "\n"

# In[19]:log = log + "cell[18];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[18];IN;model;" + print_info(model) + "\n"
log = log + "cell[18];IN;mean_absolute_percent_error;" + print_info(mean_absolute_percent_error) + "\n"
log = log + "cell[18];IN;y_test;" + print_info(y_test) + "\n"



y_pred = model.predict(X_test)
mape = mean_absolute_percent_error(y_test, y_pred)
print(f"MAPE: {mape:.1f}%")

log = log + "cell[18];OUT;X_test;" + print_info(X_test) + "\n"

# In[20]:log = log + "cell[19];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[19];IN;model;" + print_info(model) + "\n"
log = log + "cell[19];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[19];IN;plot_predictions;" + print_info(plot_predictions) + "\n"



y_pred = model.predict(X_test)
plot_predictions(y_test, y_pred)


# ## Selecting the most important variables
log = log + "cell[19];OUT;X_test;" + print_info(X_test) + "\n"
log = log + "cell[19];OUT;y_test;" + print_info(y_test) + "\n"

# In[21]:log = log + "cell[20];IN;model;" + print_info(model) + "\n"
log = log + "cell[20];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[20];IN;X_test;" + print_info(X_test) + "\n"



from sklearn.inspection import permutation_importance


pi = permutation_importance(model, X_test, y_test, n_repeats=5,
                            random_state=42, n_jobs=2)

log = log + "cell[20];OUT;pi;" + print_info(pi) + "\n"
log = log + "cell[20];OUT;y_test;" + print_info(y_test) + "\n"
log = log + "cell[20];OUT;X_test;" + print_info(X_test) + "\n"

# In[22]:log = log + "cell[21];IN;pi;" + print_info(pi) + "\n"
log = log + "cell[21];IN;i;" + print_info(i) + "\n"
log = log + "cell[21];IN;df;" + print_info(df) + "\n"
log = log + "cell[21];IN;plt;" + print_info(plt) + "\n"



sorted_idx = pi.importances_mean.argsort()
most_important_idx = [
    i for i in sorted_idx
    if pi.importances_mean[i] - 4 * pi.importances_std[i] > 0
]
most_important_names = df.columns[most_important_idx]
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(pi.importances[most_important_idx].T,
           vert=False, labels=most_important_names)
ax.set_title("Permutation Importances (test set)");


# ## Retraining a simpler model on the most important features only
log = log + "cell[21];OUT;most_important_names;" + print_info(most_important_names) + "\n"

# In[23]:log = log + "cell[22];IN;most_important_names;" + print_info(most_important_names) + "\n"



feature_subset = most_important_names.tolist()
if "Latitude" not in feature_subset:
    feature_subset += ["Latitude"]
if "Longitude" not in feature_subset:
    feature_subset += ["Longitude"]

log = log + "cell[22];OUT;feature_subset;" + print_info(feature_subset) + "\n"

# In[24]:log = log + "cell[23];IN;numeric_columns;" + print_info(numeric_columns) + "\n"
log = log + "cell[23];IN;df;" + print_info(df) + "\n"
log = log + "cell[23];IN;feature_subset;" + print_info(feature_subset) + "\n"



len(numeric_columns(df[feature_subset]))


# In[25]:log = log + "cell[24];IN;caterogical_columns;" + print_info(caterogical_columns) + "\n"
log = log + "cell[24];IN;df;" + print_info(df) + "\n"
log = log + "cell[24];IN;feature_subset;" + print_info(feature_subset) + "\n"



len(caterogical_columns(df[feature_subset]))


# In[26]:log = log + "cell[25];IN;X_train;" + print_info(X_train) + "\n"
log = log + "cell[25];IN;feature_subset;" + print_info(feature_subset) + "\n"



X_train[feature_subset]


# In[27]:log = log + "cell[26];IN;df;" + print_info(df) + "\n"
log = log + "cell[26];IN;c;" + print_info(c) + "\n"
log = log + "cell[26];IN;caterogical_columns;" + print_info(caterogical_columns) + "\n"
log = log + "cell[26];IN;feature_subset;" + print_info(feature_subset) + "\n"
log = log + "cell[26];IN;OrdinalEncoder;" + print_info(OrdinalEncoder) + "\n"
log = log + "cell[26];IN;ColumnTransformer;" + print_info(ColumnTransformer) + "\n"
log = log + "cell[26];IN;numeric_columns;" + print_info(numeric_columns) + "\n"
log = log + "cell[26];IN;HistGradientBoostingRegressor;" + print_info(HistGradientBoostingRegressor) + "\n"
log = log + "cell[26];IN;Pipeline;" + print_info(Pipeline) + "\n"
log = log + "cell[26];IN;y_train;" + print_info(y_train) + "\n"





categories = [df[c].unique()
              for c in caterogical_columns(df[feature_subset])]
ord_encoder =  OrdinalEncoder(categories=categories)

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

_= reduced_model.fit(X_train[feature_subset], y_train)

log = log + "cell[26];OUT;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[26];OUT;y_train;" + print_info(y_train) + "\n"

# In[28]:log = log + "cell[27];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[27];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[27];IN;X_train;" + print_info(X_train) + "\n"
log = log + "cell[27];IN;feature_subset;" + print_info(feature_subset) + "\n"




_ = reduced_model.fit(X_train[feature_subset], y_train)

log = log + "cell[27];OUT;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[27];OUT;y_train;" + print_info(y_train) + "\n"

# In[29]:log = log + "cell[28];IN;reduced_model;" + print_info(reduced_model) + "\n"



reduced_model[-1].n_iter_


# In[30]:log = log + "cell[29];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[29];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[29];IN;X_train;" + print_info(X_train) + "\n"
log = log + "cell[29];IN;feature_subset;" + print_info(feature_subset) + "\n"



r2_score_train = reduced_model.score(X_train[feature_subset], y_train)
print(f"r2 score (train): {r2_score_train:.3f}")

log = log + "cell[29];OUT;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[29];OUT;y_train;" + print_info(y_train) + "\n"

# In[31]:log = log + "cell[30];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[30];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[30];IN;feature_subset;" + print_info(feature_subset) + "\n"



test_r2_score = reduced_model.score(X_test[feature_subset], y_test)
print(f"r2 score (test): {test_r2_score:.3f}")

log = log + "cell[30];OUT;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[30];OUT;y_test;" + print_info(y_test) + "\n"

# In[32]:log = log + "cell[31];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[31];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[31];IN;feature_subset;" + print_info(feature_subset) + "\n"
log = log + "cell[31];IN;mean_absolute_percent_error;" + print_info(mean_absolute_percent_error) + "\n"
log = log + "cell[31];IN;y_test;" + print_info(y_test) + "\n"



y_pred = reduced_model.predict(X_test[feature_subset])
mape = mean_absolute_percent_error(y_test, y_pred)
print(f"MAPE: {mape:.1f}%")

log = log + "cell[31];OUT;y_pred;" + print_info(y_pred) + "\n"
log = log + "cell[31];OUT;reduced_model;" + print_info(reduced_model) + "\n"

# In[33]:log = log + "cell[32];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[32];IN;y_pred;" + print_info(y_pred) + "\n"
log = log + "cell[32];IN;plot_predictions;" + print_info(plot_predictions) + "\n"



plot_predictions(y_test, y_pred)


# ## Model inspection
log = log + "cell[32];OUT;y_test;" + print_info(y_test) + "\n"

# In[34]:log = log + "cell[33];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[33];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[33];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[33];IN;feature_subset;" + print_info(feature_subset) + "\n"
log = log + "cell[33];IN;np;" + print_info(np) + "\n"
log = log + "cell[33];IN;plt;" + print_info(plt) + "\n"



from sklearn.inspection import permutation_importance


pi = permutation_importance(reduced_model, X_test[feature_subset], y_test, n_repeats=10,
                            random_state=42, n_jobs=2)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(feature_subset)[sorted_idx]
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(pi.importances[sorted_idx].T,
           vert=False, labels=sorted_names)
ax.set_title("Permutation Importances (test set)");

log = log + "cell[33];OUT;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[33];OUT;y_test;" + print_info(y_test) + "\n"

# In[35]:


# %pip install -q git+https://github.com/slundberg/shap


# In[36]:log = log + "cell[35];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[35];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[35];IN;feature_subset;" + print_info(feature_subset) + "\n"
log = log + "cell[35];IN;caterogical_columns;" + print_info(caterogical_columns) + "\n"
log = log + "cell[35];IN;numeric_columns;" + print_info(numeric_columns) + "\n"
log = log + "cell[35];IN;pd;" + print_info(pd) + "\n"



import shap
shap.initjs()

explainer = shap.TreeExplainer(reduced_model[-1])
X_test_encoded = reduced_model[0].transform(X_test[feature_subset])
shap_values = explainer.shap_values(X_test_encoded)
feature_names = caterogical_columns(X_test[feature_subset]).tolist()
feature_names += numeric_columns(X_test[feature_subset]).tolist()
X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)
shap.summary_plot(shap_values, X_test_encoded, plot_type="bar")

log = log + "cell[35];OUT;shap_values;" + print_info(shap_values) + "\n"
log = log + "cell[35];OUT;X_test_encoded;" + print_info(X_test_encoded) + "\n"
log = log + "cell[35];OUT;shap;" + print_info(shap) + "\n"
log = log + "cell[35];OUT;explainer;" + print_info(explainer) + "\n"
log = log + "cell[35];OUT;pd;" + print_info(pd) + "\n"

# In[37]:log = log + "cell[36];IN;shap_values;" + print_info(shap_values) + "\n"
log = log + "cell[36];IN;X_test_encoded;" + print_info(X_test_encoded) + "\n"
log = log + "cell[36];IN;shap;" + print_info(shap) + "\n"



shap.summary_plot(shap_values, X_test_encoded, plot_size=(14, 7))

log = log + "cell[36];OUT;X_test_encoded;" + print_info(X_test_encoded) + "\n"

# In[38]:log = log + "cell[37];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[37];IN;shap;" + print_info(shap) + "\n"
log = log + "cell[37];IN;X_test_encoded;" + print_info(X_test_encoded) + "\n"
log = log + "cell[37];IN;explainer;" + print_info(explainer) + "\n"



sample_idx = 0
print("True sale price:", y_test.iloc[sample_idx])
shap.force_plot(explainer.expected_value, shap_values[sample_idx, :],
                X_test_encoded.iloc[sample_idx, :])


# In[39]:log = log + "cell[38];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[38];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[38];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[38];IN;feature_subset;" + print_info(feature_subset) + "\n"



from sklearn.inspection import plot_partial_dependence

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model, X_test[feature_subset],
                        ["Year_Built"], grid_resolution=20, ax=ax);

log = log + "cell[38];OUT;plot_partial_dependence;" + print_info(plot_partial_dependence) + "\n"
log = log + "cell[38];OUT;reduced_model;" + print_info(reduced_model) + "\n"

# In[40]:log = log + "cell[39];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[39];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[39];IN;plot_partial_dependence;" + print_info(plot_partial_dependence) + "\n"
log = log + "cell[39];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[39];IN;feature_subset;" + print_info(feature_subset) + "\n"



fig, ax = plt.subplots(figsize=(10, 5))
plot_partial_dependence(reduced_model, X_test[feature_subset], ["Gr_Liv_Area"],
                        grid_resolution=20, ax=ax);

log = log + "cell[39];OUT;reduced_model;" + print_info(reduced_model) + "\n"

# In[41]:log = log + "cell[40];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[40];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[40];IN;plot_partial_dependence;" + print_info(plot_partial_dependence) + "\n"
log = log + "cell[40];IN;X;" + print_info(X) + "\n"
log = log + "cell[40];IN;feature_subset;" + print_info(feature_subset) + "\n"



fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model, X[feature_subset], [["Gr_Liv_Area", "Year_Built"]],
                        grid_resolution=20, contour_kw={"alpha": 0.8}, ax=ax);

log = log + "cell[40];OUT;reduced_model;" + print_info(reduced_model) + "\n"

# In[42]:log = log + "cell[41];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[41];IN;reduced_model;" + print_info(reduced_model) + "\n"
log = log + "cell[41];IN;plot_partial_dependence;" + print_info(plot_partial_dependence) + "\n"
log = log + "cell[41];IN;X;" + print_info(X) + "\n"
log = log + "cell[41];IN;feature_subset;" + print_info(feature_subset) + "\n"



fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title("Partial Dependence")
plot_partial_dependence(reduced_model, X[feature_subset],
                        [["Longitude", "Latitude"]],
                        percentiles=(0., 1.),
                        grid_resolution=20, contour_kw={"alpha": 0.8}, ax=ax)
ax = fig.gca()
ax.set_xlim(X["Longitude"].min(), X["Longitude"].max())
ax.set_ylim(X["Latitude"].min(), X["Latitude"].max())
ax.set_aspect("equal")
ax.scatter(X["Longitude"], X["Latitude"]);


# ## Model selection: hyperparameter tuning

# In[43]:log = log + "cell[42];IN;HistGradientBoostingRegressor;" + print_info(HistGradientBoostingRegressor) + "\n"



HistGradientBoostingRegressor()


# In[44]:log = log + "cell[43];IN;df;" + print_info(df) + "\n"
log = log + "cell[43];IN;c;" + print_info(c) + "\n"
log = log + "cell[43];IN;caterogical_columns;" + print_info(caterogical_columns) + "\n"
log = log + "cell[43];IN;OrdinalEncoder;" + print_info(OrdinalEncoder) + "\n"
log = log + "cell[43];IN;ColumnTransformer;" + print_info(ColumnTransformer) + "\n"
log = log + "cell[43];IN;numeric_columns;" + print_info(numeric_columns) + "\n"
log = log + "cell[43];IN;HistGradientBoostingRegressor;" + print_info(HistGradientBoostingRegressor) + "\n"
log = log + "cell[43];IN;Pipeline;" + print_info(Pipeline) + "\n"



from sklearn.model_selection import RandomizedSearchCV
import numpy as np

categories = [df[c].unique()
              for c in caterogical_columns(df)]
ord_encoder =  OrdinalEncoder(categories=categories)

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
search = RandomizedSearchCV(model, params, n_iter=200, cv=3,
                            n_jobs=4, verbose=1)

log = log + "cell[43];OUT;np;" + print_info(np) + "\n"
log = log + "cell[43];OUT;model;" + print_info(model) + "\n"

# In[45]:


# _ = search.fit(X_train, y_train)


# In[46]:


# cv_results = pd.DataFrame(search.cv_results_)
# cv_results = cv_results.sort_values("mean_test_score", ascending=False)
# cv_results.to_json("ames_gbrt_search_results.json")


# In[47]:log = log + "cell[46];IN;pd;" + print_info(pd) + "\n"



cv_results = pd.read_json("ames_gbrt_search_results.json")

log = log + "cell[46];OUT;cv_results;" + print_info(cv_results) + "\n"

# In[48]:


def rename_param(column_name):
    if "__" in column_name:
        return column_name.rsplit("__", 1)[1]
    return column_name

log = log + "cell[47];OUT;rename_param;" + print_info(rename_param) + "\n"

# In[49]:log = log + "cell[48];IN;cv_results;" + print_info(cv_results) + "\n"
log = log + "cell[48];IN;rename_param;" + print_info(rename_param) + "\n"



cv_results.rename(rename_param, axis=1).head(5)


# ### Interactions between hyperparameters and generalization

# In[73]:log = log + "cell[49];IN;cv_results;" + print_info(cv_results) + "\n"
log = log + "cell[49];IN;rename_param;" + print_info(rename_param) + "\n"
log = log + "cell[49];IN;np;" + print_info(np) + "\n"



import plotly.express as px


fig = px.parallel_coordinates(
    cv_results.rename(rename_param, axis=1).apply({
        "learning_rate": np.log10,
        "max_leaf_nodes": np.log2,
        "max_bins": np.log2,
        "mean_test_score": lambda x: x,
    }),
    color="mean_test_score",
    color_continuous_scale=px.colors.diverging.Portland,
)
fig.show()


# Let's zoom on the top performing models by using the `query` methods of the dataframe. Note that the axis have a narrower range now:
log = log + "cell[49];OUT;px;" + print_info(px) + "\n"

# In[74]:log = log + "cell[50];IN;px;" + print_info(px) + "\n"
log = log + "cell[50];IN;cv_results;" + print_info(cv_results) + "\n"
log = log + "cell[50];IN;rename_param;" + print_info(rename_param) + "\n"
log = log + "cell[50];IN;np;" + print_info(np) + "\n"



fig = px.parallel_coordinates(
    cv_results.rename(rename_param, axis=1).apply({
        "learning_rate": np.log10,
        "max_leaf_nodes": np.log2,
        "max_bins": np.log2,
        "mean_test_score": lambda x: x,
    }).query("mean_test_score > 0.88"),
    color="mean_test_score",
    color_continuous_scale=px.colors.diverging.Portland,
)
fig.show()


# Let's check that the inner CV scores still approximately reflect the true generatlization score measured on held out data even when we select the best model from hundred of possible candidates via random search:

# In[96]:log = log + "cell[51];IN;cv_results;" + print_info(cv_results) + "\n"



best_search_result = cv_results.nlargest(n=1, columns=["mean_test_score"]).iloc[0]
print(f'R2 score of best candidate (inner CV): {best_search_result["mean_test_score"]:.3f}'
      f' (+/-{3 * best_search_result["std_test_score"]:.3f})')

log = log + "cell[51];OUT;best_search_result;" + print_info(best_search_result) + "\n"

# In[97]:log = log + "cell[52];IN;model;" + print_info(model) + "\n"
log = log + "cell[52];IN;best_search_result;" + print_info(best_search_result) + "\n"
log = log + "cell[52];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[52];IN;X_train;" + print_info(X_train) + "\n"
log = log + "cell[52];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[52];IN;X_test;" + print_info(X_test) + "\n"



model.set_params(**best_search_result["params"])
model.fit(X_train, y_train)
held_out_score = model.score(X_test, y_test)
print(f'R2 score of best candidate on held-out data: {held_out_score:.3f}')


# ### Exploration of the accuracy / prediction latency trade-off

# In[52]:log = log + "cell[53];IN;cv_results;" + print_info(cv_results) + "\n"



cv_results["safe_test_score"] = cv_results["mean_test_score"] - cv_results["std_test_score"]

log = log + "cell[53];OUT;cv_results;" + print_info(cv_results) + "\n"

# In[53]:log = log + "cell[54];IN;c;" + print_info(c) + "\n"
log = log + "cell[54];IN;cv_results;" + print_info(cv_results) + "\n"



import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()


param_names = [c for c in cv_results.columns
               if c.startswith("param_")]
fig = px.scatter(cv_results, x="mean_score_time", y="safe_test_score",
                 hover_data=param_names)

fig.show()


# In[ ]:




f = open(os.path.join(my_dir_path, "ames_housing_m_log.txt"), "w")
f.write(log)
f.close()
