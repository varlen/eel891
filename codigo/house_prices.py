import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

# Dataframe de treino
df_train = pd.read_csv('./dataset/train.csv')

# Observação estatistica dos dados
for col in df_train.columns:
    print(df_train[col].describe())

# Scatterplots das variaveis numericas
variaveis_numericas = [ col 
                       for col in df_train.columns 
                       if df_train[col].describe().dtype == 'float64' 
                       and not col in ['Id','SalePrice'] ]
for var in variaveis_numericas :
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    
# Scatterplots razoavelmente correlacionados em relação a SalePrice
lin_cols = ['LotFrontage','YearBuilt','TotalBsmtSF',
            '1stFlrSF', 'GrLivArea','GarageArea','WoodDeckSF']