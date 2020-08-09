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

# # Subroutine 1
# Generate points based on the labels (taken as an input from the GUI)
# structure of the point:
#     (t0, vel, flag (1=hyperbolic, 0=linear), label)

# In[1]:


import random
import numpy as np
import matplotlib.pylab as plt
from scipy import signal



def points_gen(nDA,nR,nCN,nMU=5):
    # The function generates random values of t0 - velocity pairs
    # based on number of requested waves:
    # nDA - number of direct waves (linear moveout, label 'D')
    # nR - number of reflections (hyperbolic moveout, label 'R')
    # nCN - number of coherent noise events (linear moveout, label 'N')
    # nMU - number of multiples (hyperbolic, label 'M')
    # outputs (nDA+nR+nCN) * (4) list
    # each point in the list has the following structure
    # [t0 (intercept time), velocity, flag(1=hyperbolic, 0=linear), label(see above))]
    
    
    # direct arrival
    direct = []
    for n in range(nDA):
        direct.append([random.uniform(0,.1),random.uniform(.5,1.5),0,'D'])
    
    reflected = []
    multiples = []
    for n in range(nR):
        reflection = [random.uniform(0,3),random.uniform(1.5,3.5),1,'R']
        reflected.append(reflection)
        for nmult in range(2,nMU):
            multiples.append([nmult * reflection[0], reflection[1],1,'M'])
        
    noise = []
    for n in range(nCN):
        noise.append([random.uniform(-3,3),random.uniform(-3,3),0,'N'])
        
    
#    for n in range(nMU):
#        noise.append([random.uniform(random.uniform(2,2),4),random.uniform(1.5,2.5),1,'M'])

    events = direct + reflected + noise + multiples
    return events


# In[2]:


def points_plot(events):
    x = [x/1000 for x in range(0,2000,25)]
    
    fig, ax = plt.subplots()
    
    # plot waves
    for i in events:
        if i[3] == 'D':
            ax.plot(x,[i[0] + offset/i[1] for offset in x],'r')
        if i[3] == 'N':
            ax.plot(x,[i[0]+offset/i[1] for offset in x],'b')
        if i[3] == 'R':
            ax.plot(x,[np.sqrt(i[0]**2 + offset**2 / i[1]**2) for offset in x],'g')
        if i[3] == 'M':
            ax.plot(x,[np.sqrt(i[0]**2 + offset**2 / i[1]**2) for offset in x],'k')
    
    plt.ylabel('Time, s')
    plt.xlabel('Offset, km')
    ax.set_xlim([0,2])
    ax.set_ylim([0,4])
    ax.invert_yaxis()
    ax.set_aspect(1)
    return ax


# In[3]:


events=points_gen(2,2,2)
ax = points_plot(events)
plt.show(ax)

my_store_info((3, 1, "events"), events)
my_store_info((3, 1, "plt"), plt)

# In[4]:


def t_linear(x, v, t):
    # return a linear event (direct or coherent noise)
    return t + x/v

def t_reflected(x,v,t):
    return np.sqrt(t**2 + x**2 / v**2)

def points_to_data(events, dx = 0.005, xmax = 2):
    x = np.arange(0, xmax + dx, dx)
    t=[]
    
    for i in events:
        if i[3] == 'D' or i[3] == 'N':
            t.append(t_linear(x, i[1], i[0]))
        if i[3] == 'R' or i[3] == 'M':
            t.append(t_reflected(x,i[1],i[0]))
    return t


# In[5]:my_store_info((5, 0, "events"), events)



data = points_to_data(events)
dataround = np.round(data,decimals=2)

my_store_info((5, 1, "dataround"), dataround)

# In[6]:


dt = 0.01
tmax = 4
t = np.arange(0,tmax + dt,dt)

dx = 0.005
xmax = 2
x = np.arange(0, xmax + dx, dx)

my_store_info((6, 1, "t"), t)
my_store_info((6, 1, "x"), x)

# In[7]:my_store_info((7, 0, "t"), t)
my_store_info((7, 0, "x"), x)
my_store_info((7, 0, "dataround"), dataround)



datamatrix = np.zeros((len(t),len(x)))

for event in dataround:
    for n, i in enumerate(event):
        idx = np.where((t >= i - 0.0001) & (t <= i + .0001))
        if np.size(idx[0]) != 0:
            #print(idx[0],n)
            datamatrix[idx[0][0]][n] = 1

my_store_info((7, 1, "datamatrix"), datamatrix)

# In[8]:my_store_info((8, 0, "datamatrix"), datamatrix)
my_store_info((8, 0, "plt"), plt)



#numpy.apply_along_axis


def conv_ricker(array):
    points = 100
    a = 4.0
    ricker = signal.ricker(points, a)
    return np.convolve(array,ricker,mode='same')

data_ricker = np.apply_along_axis(conv_ricker, 0, datamatrix)

import numpy as np

plt.imshow(datamatrix, cmap='gray', interpolation='bicubic', extent=[0,2,4,0]);
plt.show()

plt.imshow(data_ricker, cmap='gray', interpolation='bicubic', extent=[0,2,4,0]);
plt.show()


# In[9]:


# Convert data in list format to dictionary
    
def makeEventsDict(events):
    eventsDict = {}
    labelsDict = {}
    
    labelsDict['label'] = []
    eventsDict['direct'], eventsDict['reflected'], eventsDict['coherentnoise'], eventsDict['multiples'] = [],[],[],[]
    eventsDict['events'] = []
    for each in events:
#         print(each)
#         print(each[0])
#         print(each[3])
#         eventsDict['direct'].append(each[0])
#         eventsDict['reflected'].append(each[1])
#         eventsDict['coherentnoise'].append(each[2])
         eventsDict['events'].append(each[0:3])
         labelsDict['label'].append(each[3])
    return(eventsDict,labelsDict)


# In[10]:my_store_info((10, 0, "events"), events)



from sklearn import svm
SVC = svm.SVC()

# This function takes a events list, turns it into two dictionaries combined, splits that into two arrays for X and Y
# trains a SVM label on them and then returns that model output details
# The model will need to be run on a input for a prediction

def comboFunctionA(events):
    testEvents = makeEventsDict(events)
    eventsDict = testEvents[0]
    labelsDict = testEvents[1]
    X = eventsDict['events']
#     print("X = ",X)
    y = labelsDict['label']
#     print("y = ",y)
    clf = svm.SVC()
    output = clf.fit(X, y)
    return(output)


# In[11]:


# training on a test set
events_train=points_gen(200,200,200,200)

tempAnswer = comboFunctionA(events_train)
tempAnswer

my_store_info((11, 1, "tempAnswer"), tempAnswer)

# In[12]:my_store_info((12, 0, "tempAnswer"), tempAnswer)



tempAnswer.predict([[.1,1.5,0]])


# In[ ]:




store_vars.append(my_labels)
f = open(os.path.join(my_dir_path, "nb_1002479_log.dat"), "wb")
pickle.dump(store_vars, f)
f.close()
