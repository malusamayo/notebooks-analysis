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

# In[1]:



import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

log = log + "cell[1];OUT;plt;" + print_info(plt) + "\n"
log = log + "cell[1];OUT;np;" + print_info(np) + "\n"
log = log + "cell[1];OUT;gaussian_kde;" + print_info(gaussian_kde) + "\n"

# In[2]:


from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=50000, n_features=20,
                                    n_informative=2, n_redundant=5,
                                    random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=500, random_state=0)


# ## Probability reliability plots
log = log + "cell[2];OUT;y_test;" + print_info(y_test) + "\n"
log = log + "cell[2];OUT;y_train;" + print_info(y_train) + "\n"

# In[3]:log = log + "cell[3];IN;y_test;" + print_info(y_test) + "\n"
log = log + "cell[3];IN;plt;" + print_info(plt) + "\n"
log = log + "cell[3];IN;np;" + print_info(np) + "\n"
log = log + "cell[3];IN;gaussian_kde;" + print_info(gaussian_kde) + "\n"



from sklearn.calibration import calibration_curve


def plot_calibration(classifiers, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(classifiers)))
    for (name, clf), c in zip(classifiers, colors):
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos =                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value =             calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % name, c=c)

        x = np.linspace(0, 1, 500)
        smoothed_prob_pos = gaussian_kde(prob_pos)(x)
        ax2.plot(x, smoothed_prob_pos, c=c)
        ax2.fill_between(x, 0, smoothed_prob_pos, alpha=0.1, facecolor=c)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("# samples")

    plt.tight_layout()

log = log + "cell[3];OUT;plot_calibration;" + print_info(plot_calibration) + "\n"
log = log + "cell[3];OUT;y_train;" + print_info(y_train) + "\n"
log = log + "cell[3];OUT;X_test;" + print_info(X_test) + "\n"
log = log + "cell[3];OUT;y_test;" + print_info(y_test) + "\n"

# In[4]:log = log + "cell[4];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
plot_calibration([('Logistic Regression', lr)])

log = log + "cell[4];OUT;lr;" + print_info(lr) + "\n"

# In[5]:log = log + "cell[5];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



from sklearn.svm import LinearSVC

svc = LinearSVC(C=1.0)
plot_calibration([('Support Vector Machine', svc)])

log = log + "cell[5];OUT;svc;" + print_info(svc) + "\n"
log = log + "cell[5];OUT;LinearSVC;" + print_info(LinearSVC) + "\n"

# In[6]:log = log + "cell[6];IN;lr;" + print_info(lr) + "\n"
log = log + "cell[6];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[6];IN;svc;" + print_info(svc) + "\n"



from sklearn.model_selection import cross_val_score

print("Logistic Regression: %0.3f" %
      cross_val_score(lr, X_train, y_train, cv=5).mean())
print("Linear SVC: %0.3f" %
      cross_val_score(svc, X_train, y_train, cv=5).mean())

log = log + "cell[6];OUT;lr;" + print_info(lr) + "\n"
log = log + "cell[6];OUT;svc;" + print_info(svc) + "\n"
log = log + "cell[6];OUT;y_train;" + print_info(y_train) + "\n"

# In[12]:log = log + "cell[7];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100)
plot_calibration([('Random Forest', rf)])

log = log + "cell[7];OUT;rf;" + print_info(rf) + "\n"

# In[13]:log = log + "cell[8];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
plot_calibration([('Naive Bayes', gnb)])

log = log + "cell[8];OUT;gnb;" + print_info(gnb) + "\n"
log = log + "cell[8];OUT;GaussianNB;" + print_info(GaussianNB) + "\n"

# In[14]:log = log + "cell[9];IN;lr;" + print_info(lr) + "\n"
log = log + "cell[9];IN;gnb;" + print_info(gnb) + "\n"
log = log + "cell[9];IN;svc;" + print_info(svc) + "\n"
log = log + "cell[9];IN;rf;" + print_info(rf) + "\n"
log = log + "cell[9];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



classifiers =  [
    ('Logistic Regression', lr),
    ('Naive Bayes', gnb),
    ('Support Vector Classification', svc),
    ('Random Forest', rf),
]

plot_calibration(classifiers)


# ## Calibration

# In[15]:log = log + "cell[10];IN;LinearSVC;" + print_info(LinearSVC) + "\n"
log = log + "cell[10];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



from sklearn.calibration import CalibratedClassifierCV

svc = LinearSVC(C=1.0)
sigmoid_svc = CalibratedClassifierCV(svc, method='sigmoid')
isotonic_svc = CalibratedClassifierCV(svc, method='isotonic')
svc_models = [
    ('SVM (raw)', svc),
    ('SVM + sigmoid calibration', sigmoid_svc),
    ('SVM + isotonic calibration', isotonic_svc),
]
plot_calibration(svc_models)

log = log + "cell[10];OUT;CalibratedClassifierCV;" + print_info(CalibratedClassifierCV) + "\n"

# In[16]:log = log + "cell[11];IN;GaussianNB;" + print_info(GaussianNB) + "\n"
log = log + "cell[11];IN;CalibratedClassifierCV;" + print_info(CalibratedClassifierCV) + "\n"
log = log + "cell[11];IN;plot_calibration;" + print_info(plot_calibration) + "\n"



gnb = GaussianNB()
sigmoid_gnb = CalibratedClassifierCV(gnb, method='sigmoid')
isotonic_gnb = CalibratedClassifierCV(gnb, method='isotonic')
gnb_models = [
    ('Naive Bayes (without calibration)', gnb),
    ('Naive Bayes + sigmoid calibration', sigmoid_gnb),
    ('Naive Bayes + isotonic calibration', isotonic_gnb),
]
plot_calibration(gnb_models)

log = log + "cell[11];OUT;gnb_models;" + print_info(gnb_models) + "\n"

# In[17]:log = log + "cell[12];IN;gnb_models;" + print_info(gnb_models) + "\n"
log = log + "cell[12];IN;y_train;" + print_info(y_train) + "\n"
log = log + "cell[12];IN;X_test;" + print_info(X_test) + "\n"
log = log + "cell[12];IN;y_test;" + print_info(y_test) + "\n"



from sklearn.metrics import log_loss, brier_score_loss

for name, model in gnb_models:
    prob_pos = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    
    bs = brier_score_loss(y_test,prob_pos)
    ll = log_loss(y_test, prob_pos)
    print("%s:\tBrier score = %0.3f, log loss = %0.3f"
          % (name, bs, ll))


# In[ ]:




f = open(os.path.join(my_dir_path, "Classifier_calibration_m_log.txt"), "w")
f.write(log)
f.close()
