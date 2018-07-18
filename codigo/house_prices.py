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
#for col in df_train.columns:
#    print(df_train[col].describe())


variaveis_a_desconsiderar = [
        
        # Numericas
       'Id','MSSubClass','SalePrice', 
       'OverallQual', 'OverallCond','YearRemodAdd',
       'BsmtFinSF2', '2ndFlrSF', 
       'LowQualFinSF', 'BsmtFullBath',
       'BsmtHalfBath','FullBath','HalfBath',
       'BedroomAbvGr','KitchenAbvGr',
       'Fireplaces','GarageCars', 'EnclosedPorch',
       '3SsnPorch','ScreenPorch','PoolArea',
       'MiscVal','MoSold','YrSold','TotRmsAbvGrd'
       
       # Categoricas
       'MSZoning','Street','LotShape','Utilities','LandSlope',
       'BldgType','Condition1','RoofStyle','Exterior1st',
       'Exterior2nd','MasVnrType','Foundation','BsmtCond','BsmtExposure',
       'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
       'Electrical','Functional','GarageType','GarageFinish','GarageQual',
       'GarageCond','PavedDrive','Fence','MiscFeature',
       'SaleType','SaleCondition','RoofMatl','Condition2','HouseStyle'
       
]

# Scatterplots das variaveis numericas
variaveis_numericas = [ col 
                       for col in df_train.columns 
                       if df_train[col].describe().dtype == 'float64' 
                       and not col in variaveis_numericas_a_desconsiderar ]

for var in variaveis_numericas :
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Boxplot das variaveis categoricas    
variaveis_categoricas = [ col for col in df_train.columns 
                         if not col in variaveis_a_desconsiderar
                         and not col in variaveis_numericas ]
variaveis_categoricas.append('OverallQual')
variaveis_categoricas.append('OverallCond')


for var in variaveis_categoricas:
    
    sns.boxplot(y = "SalePrice", x = var, data = df_train)
    sns.stripplot(y = "SalePrice", x = var, data = df_train)
    plt.show()
    
    
# Scatterplots razoavelmente correlacionados em relação a SalePrice
lin_cols = ['LotFrontage','YearBuilt','TotalBsmtSF',
            '1stFlrSF', 'GrLivArea','GarageArea','WoodDeckSF']


# Matriz de correlacao 
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, aax = plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



