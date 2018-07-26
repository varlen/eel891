import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

# Dataframe de treino
df_train_original = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')

train_len = len(df_train_original)

df_total = pd.concat([df_train_original,df_test])

#### Pre-processamento

# Remover a coluna de Id
df_train = df_total.drop("Id",axis=1)

# Ver as informações do dataFrame
df_train.info()

# Verificarr dados faltantes
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Para algumas variáveis, a descrição fornece um insight de como
# completar os dados faltantes.
default_values = {
    "Alley" : "--",
    "LotFrontage" : 0,
    "BedroomAbvGr" : 0,
    "ExterCond" : "TA",
    "ExterQual" : "TA",
    "BsmtQual" : "--",
    "BsmtCond" : "--",
    "BsmtExposure" : "--",
    "BsmtFinType1" : "--",
    "BsmtFinType2" : "--",
    "BsmtHalfBath" : 0,
    "BsmtFullBath" : 0,
    "BsmtFinSF1" : 0,
    "TotalBsmtSF": 0,
    "GarageCars": 0,
    "GarageArea": 0,
    "SaleType": "Oth",
    "BsmtUnfSF" : 0,
    "BsmtFinSF2" : 0,
    "FireplaceQu" : "--",
    "PoolQC" : "--",
    "GarageFinish" : "--",
    "GarageQual" : "--",
    "GarageCond" : "--",
    "GarageType" : "No",
    "KitchenQual" : "TA",
    "PavedDrive" : "N",
    "MasVnrType" : "None",
    "MasVnrArea" : 0,
    "MiscFeature" : "--",
    "MiscVal" : 0,
    "SaleCondition" : "Normal",
    "TotRmsAbvGrd" : 1,
    "Utilities" : "AllPub",
    "Functional" : "Typ",
    "Electrical" : "Mix",
    "Exterior2nd" : "Other",
    "Exterior1st": "Other"
}

for feature,fill in default_values.items():
    df_train.loc[:, feature] = df_train.loc[:, feature].fillna(fill)
 
## Algumas variaveis do dataset parecem numéricas mas são na verdade categoricas.
## Usando os dados no arquivo data_description.txt para obter as variaveis categoricas.
df_train.replace( { "MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                   50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                   80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                   150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"}
                                   }, inplace = True)

# Outra variavel que parece numérica porém é categórica, é a variavel MoSold.
# Entretanto, pela sua natureza, acredito que não seja relevante na predição.
# Para confirmar essa hipotese, verifica-se o scatterplot dela vs preços.

data = pd.concat([df_train['SalePrice'], df_train['MoSold']], axis=1)
data.plot.scatter(x='MoSold', y='SalePrice')
plt.show()
sns.countplot(data['MoSold'])

# MiscFeature também é uma variavel passivel de eliminação, pois não aparece
# em quase nenhum item do conjunto de dados.
data = pd.concat([df_train['SalePrice'], df_train['MiscFeature']], axis=1)
sns.boxplot(x='MiscFeature', y='SalePrice', data = df_train)
plt.show()
sns.countplot(data['MiscFeature'])

data = pd.concat([df_train['SalePrice'], df_train['MiscVal']], axis=1)
data.plot.scatter(x='MiscVal', y='SalePrice')
plt.show()
sns.countplot(data['MiscVal'])
df_train.drop("MiscVal", axis=1, inplace=True)

# De acordo com os gráficos, o período do ano com maior vendas
#  é também o período com maior variancia de preços.
# Assim, elimina-se MoSold do dataset sem culpa
df_train.drop("MoSold", axis=1, inplace=True)

# Observando a descrição do dataset para a coluna Fence levanta a suspeita de 
# que esta não forneça informações relevantes. Pela contagem de valores mostrada
# pelas informações do dataset, decidi eliminar esta coluna
df_train.drop("Fence", axis=1, inplace=True)

# GarageYrBlt também será desconsiderada, ja que as informações do ano de 
# construção e de garagem existem separadamente em outras variaveis
df_train.drop("GarageYrBlt", axis=1, inplace=True)

# Com isso, ainda resta tratar 4 valores faltantes em MSZoning
# Vejamos as informações dessa feature
data = pd.concat([df_train['SalePrice'], df_train['MSZoning']], axis=1)
sns.boxplot(x='MSZoning', y='SalePrice', data = df_train)
plt.show()
sns.countplot(data['MSZoning'])
# Observando os graficos, observa-se que não existe informação para zona A.
# Sendo assim, assumi que os valores faltantes na verdade são de zona A.
df_train.loc[:, 'MSZoning'] = df_train.loc[:, 'MSZoning'].fillna('A')

# Algumas variaveis são categoricas, porém as categorias estão ordenadas.
# Para estas variaveis, as categorias serão mapeadas em graus numéricos.
# Consideraram-se para tal, features as quais acreditei exercerem influência
# proporcional no preço final de acordo com a categoria. 
category_grade_mapping = {
    "Alley" : {
        "--" : 0, "Grvl" : 1, "Pave" : 2
    },
    "ExterQual" : {
        "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5 
    },
    "ExterCond" : {
        "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5 
    },
    "BsmtQual" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "BsmtCond" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "BsmtExposure" : {
        "--": 0, "No" : 1, "Mn" : 2, "Av" : 3, "Gd" : 4
    },
    "BsmtFinType1" : {
        "--": 0, "Unf" : 1, "LwQ" : 2, 
        "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6
    },
    "BsmtFinType2" : {
        "--": 0, "Unf" : 1, "LwQ" : 2, 
        "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6
    },
    "HeatingQC" : {
        "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "CentralAir" : {
        "N" : 0, "Y" : 1
    },
    "KitchenQual" : {
        "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "FireplaceQu" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "GarageFinish" : {
        "--" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3
    },
    "GarageQual" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "GarageCond" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "PavedDrive" : {
        "N" : 0, "P" : 1, "Y" : 2
    },
    "PoolQC" : {
        "--": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5
    },
    "Utilities" : {
        "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4
    }
}
df_train.replace(category_grade_mapping, inplace = True)

# Scatterplots das variaveis numericas
var_cat = [ col 
            for col in df_train.columns 
            if df_train[col].describe().dtype == 'object']

for var in var_num :
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Boxplot das variaveis categoricas    
var_num = [ col 
            for col in df_train.columns 
            if df_train[col].describe().dtype != 'object']

for var in var_cat:
    sns.boxplot(y = "SalePrice", x = var, data = df_train)
    sns.stripplot(y = "SalePrice", x = var, data = df_train)
    plt.show()

# Mapa de calor 
corrmat = df_train.corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, aax = plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Colunas mais correlacionadas
print(cols)

# O mapa de calor mostra as variaveis numericas OverallQual, GrLivArea, ExterQual, 
# KitchenQual, 'GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF e BsmtQual 
# como mais correlacionadas ao preço de venda da casa
for var in cols :
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
# Gerar dummies das categoricas
df_train = pd.get_dummies(df_train)
    
# 1o treinamento
from sklearn.ensemble import RandomForestRegressor

train_set = df_train[:train_len]
X_train, X_test, Y_train, Y_test = train_test_split(
        train_set.drop('SalePrice',axis=1), train_set.SalePrice)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, Y_train)

scr_tr = rfr.score(X_train, Y_train)
scr_ts = rfr.score(X_test, Y_test)

result = rfr.predict(df_train[train_len:].drop('SalePrice', axis=1))

