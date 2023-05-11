import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #
#import pandas as pd
import io
#import numpy as np
#from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC