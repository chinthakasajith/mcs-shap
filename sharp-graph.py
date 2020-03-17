import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
# df = pd.read_csv('/home/chinthaka/PycharmProjects/sharp/pca-analysis.csv')
df = pd.read_csv('/home/chinthaka/PycharmProjects/sharp/winequality-red.csv')
# Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
# The target variable is 'quality'.
# Y = df['Seriour Drug Adverse Event']
# X =  df[['Seriour Drug Adverse Event','Adverse Event','Allegation of Death','Lack of Efficacy','Product Quality Complaint','Off-Label Usage','Recreational drug use or abuse','Off-labelusagthatrequirsHCPadvc','ImproperAttributionofPrdctBnfts','Taste (negative)','On Label Incomplete','Competitive Comparative Threat','CnsmrQstn/Sggstn(NtChnnl-Spcfc)','ConsumrQstn/Sggstn(Chnnl-Spcfc)','Product cost','Political','Positive Comment','Negative Comment','Not Otherwise Classified','Offensive or Abusive Language','Spam/Nonsense','Tagging','Sticker Emoji']]


Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

rf = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
rf.fit(X_train, Y_train)
# print(rf.feature_importances_)
# importances = rf.feature_importances_
# indices = np.argsort(importances)
# features = X_train.columns
#
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importaggggnce')
# plt.show()

# The summary plot
# import shap
# rf_shap_values = shap.KernelExplainer(rf.predict,X_test)
# print(rf.predict)
#
# shap.summary_plot(rf_shap_values, X_test)

import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")