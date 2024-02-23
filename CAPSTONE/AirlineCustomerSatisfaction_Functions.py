import pandas as pd
from pandas import DataFrame as df
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#!pip install --user pandas_profiling
#!pip install --user markupsafe==2.0.1
#!pip install pandasql
import pandas_profiling
import sweetviz as sv
from pandasql import sqldf
np.seterr(under='ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
from scipy import stats
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
#!pip install --upgrade --user threadpoolctl
#!pip install --upgrade --user numpy
#!pip install --user numpy==1.21.4
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


# Phase 2
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
#from yellowbrick.model_selection import RFECV

import warnings
from scipy.stats import kurtosis

from numpy import percentile


import pickle
import sys
import os
import gc
import traceback

import sweetviz as sv


import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.model_selection import FeatureImportances

from collections import Counter
from numpy import where

import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

from os import chdir
from dateutil.parser import parse
# Systematic: Components of the time series that have consistency or recurrence and can be described and modeled.
# Non-Systematic: Components of the time series that cannot be directly modeled.
import pmdarima as pm
from pmdarima.utils import array
from dfply import *
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA

# Level: The average value in the series.
# Trend: The increasing or decreasing value in the series.
# Seasonality: The repeating short-term cycle in the series.
# Noise: The random variation in the series.


from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dython.nominal import associations
from plotnine import *
import category_encoders as ce

#from keras.callbacks import ModelCheckpoint
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten, Dropout, Input
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import Adam, SGD
#from keras.constraints import maxnorm

import numpy as np
import sys
import os
import gc
import traceback
import missingno as msno
#from missingpy import MissForest
from impyute.imputation.cs import mice

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, roc_curve
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, RandomizedSearchCV, KFold

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import signal

#from keras.models import Sequential
#import keras.backend as K

#from keras.layers import Dense, Input, Dropout, LSTM
#from keras.optimizers import SGD, Adam
#from keras.models import Model
#from keras.models import load_model
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import EarlyStopping
#from keras.utils.vis_utils import plot_model

from xgboost.sklearn import XGBRegressor

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, IsolationForest, StackingClassifier
from sklearn.model_selection import learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import itertools
from xgboost.sklearn import XGBClassifier
#from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics, datasets, linear_model

from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

from sklearn.feature_selection import RFECV
#from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from mlens.visualization import corrmat

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
plt.style.use('seaborn')



def dropColumns(data, columnsList):
    data = data.drop(columnsList, axis=1)
    return data

def renameAllColumns(data, columnNamesList):
    data.columns = columnNamesList
    return data
    
def performLabelEncoding(data, columnsList, displayEncodedData=False, displayInfo=False):
    labelencoder = LabelEncoder()
    for col in columnsList:
        data[col+'_Encoded']=labelencoder.fit_transform(data[col])
    # Display previous and encoded values
    if displayEncodedData==True:
        for col in columnsList:
            query="select distinct "+col+", "+col+"_Encoded from data"
            #print(query)
            temp=sqldf(query) 
            display(temp)

    # Drop pre-encoded columns
    data=dropColumns(data,columnsList)

    # Rename encoded column names to previous column names
    data.columns = data.columns.str.replace('_Encoded', '')
    if displayInfo==True:
        data.info()
    return data
    
# Perform Min-Max Scaling
def performMinMaxScaling(data):
    d1 = data.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(d1)
    scaledData = pd.DataFrame(scaledData, columns=d1.columns, index=d1.index)
    return scaledData

# Perform scaling and return scaled data
def scaleData(data, targetColumn):
    d2 = dropColumns(data,[targetColumn])
    d1Scaled = performMinMaxScaling(d2)
    data=data[[targetColumn]]
    data=pd.concat([data.loc[:, [targetColumn]], d1Scaled], axis=1)
    return data
    
def DBScanOutlier(data):
    minSamples=data.shape[1]+1
    d1 = data.copy()

    scaler = MinMaxScaler()
    d2 = scaler.fit_transform(d1)
    d2 = pd.DataFrame(d2, columns=d1.columns)
    db = DBSCAN(eps=0.85, min_samples=minSamples).fit(d2)

    #from sklearn import metrics
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    outlierRow = []
    for i in range(len(labels)):
        if (labels[i] == -1):
            # print(list((i, labels[i])))
            outlierRow.append(i)

    #level0Outlier, level1Outlier = data.loc[outlierRow, response].value_counts().sort_index().tolist()
    
    return outlierRow#, level0Outlier, level1Outlier
    
def performPCA(data,targetColumn):
    pcaObj = pca()
    pcaData = pcaObj.createPCA(data, targetColumn)
    pcaData = pd.concat([pcaData.iloc[:, 0:15], pcaData[targetColumn]], axis=1)
    return pcaData
    
def performBinning(data, columnsToBin):
    binnedData=data.copy()
    binnedData = createBin(binnedData, columnsToBin, 7)
    
    binnedData=dropColumns(binnedData,columnsToBin)
    binnedData.columns = binnedData.columns.str.replace('_BINNED', '')
    binnedData[columnsToBin]=binnedData[columnsToBin].astype('object')
    binnedData = performLabelEncoding(binnedData, columnsToBin, False, False)
    return binnedData

def createBin(data, numericCols, binNumber=7):
    d1 = data.copy()
    for i in numericCols[0:len(numericCols)]:
        bins = np.linspace(d1[i].min(), d1[i].max(), binNumber)
        d1[i + "_BINNED"] = pd.cut(d1[i], bins, precision=1, include_lowest=True, right=True)
    return d1
