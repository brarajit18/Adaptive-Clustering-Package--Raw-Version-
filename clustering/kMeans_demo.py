# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../datasets/Social_Network_Ads.csv')

df1 = df[['Age', 'EstimatedSalary']]

sns.scatterplot(x=df1['Age'], y=df1['EstimatedSalary'])