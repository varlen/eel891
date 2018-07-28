import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, RobustScaler
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
missing_data.head(10)

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
# De acordo com os gráficos, o período do ano com maior vendas
#  é também o período com maior variancia de preços.
# Assim, elimina-se MoSold do dataset sem culpa
df_train.drop("MoSold", axis=1, inplace=True)

# MiscFeature também é uma variavel passivel de eliminação, pois não aparece
# em quase nenhum item do conjunto de dados.
data = pd.concat([df_train['SalePrice'], df_train['MiscFeature']], axis=1)
sns.boxplot(x='MiscFeature', y='SalePrice', data = df_train)
plt.show()
sns.countplot(data['MiscFeature'])
df_train.drop("MiscFeature", axis=1, inplace=True)

data = pd.concat([df_train['SalePrice'], df_train['MiscVal']], axis=1)
data.plot.scatter(x='MiscVal', y='SalePrice')
plt.show()
sns.countplot(data['MiscVal'])
df_train.drop("MiscVal", axis=1, inplace=True)


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

data = pd.concat([df_train['SalePrice'], df_train['MSZoning']], axis=1)
sns.boxplot(x='MSZoning', y='SalePrice', data = df_train)
plt.show()
sns.countplot(data['MSZoning'])

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


var_cat = [ col 
            for col in df_train.columns 
            if df_train[col].describe().dtype == 'object']

var_num = [ col 
            for col in df_train.columns 
            if df_train[col].describe().dtype != 'object']

# Scatterplots das variaveis numericas
#for var in var_num :
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))   

# Boxplot das variaveis categoricas 
#for var in var_cat:
#    sns.boxplot(y = "SalePrice", x = var, data = df_train)
#    sns.stripplot(y = "SalePrice", x = var, data = df_train)
#    plt.show()

# Mapa de calor 
corrmat = df_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True,)

corrmat = df_train[:train_len].corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, aax = plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Colunas mais correlacionadas
print(cols)

# Criando variaveis artificiais a partir do dataset
#df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF'] + df_train['LotArea']
df_train['GarageScore'] = df_train['GarageArea'] * df_train['GarageCars'] * df_train['GarageCond'] * df_train['GarageQual']
#df_train['OverallScore'] = df_train['OverallCond'] * df_train['OverallQual']
#df_train['ExternalScore'] = df_train['ExterQual'] * df_train['ExterCond']
#df_train['BasementScore'] = df_train['BsmtQual'] * df_train['BsmtCond']
#df_train['RoomSpace'] = (df_train['1stFlrSF'] + df_train['2ndFlrSF']) / df_train['TotRmsAbvGrd']
df_train['IsFunctional'] = df_train['Functional']
df_train.replace({ 'IsFunctional' : { 'Typ':1, 'Min1':1, 'Min2':1, 'Mod': 1, 
                                     'Maj1':0, 'Maj2':0, 'Sev':0, 'Sal':0}}, inplace = True)

# Removendo variaveis que não acrescentam informação
df_train.drop('GarageQual',axis=1, inplace = True)
df_train.drop('Utilities',axis=1, inplace = True)
df_train.drop('GarageCars',axis=1, inplace = True)
df_train.drop('YrSold',axis=1, inplace = True)
df_train.drop('YearRemodAdd',axis=1, inplace = True)
df_train.drop('PavedDrive',axis=1, inplace = True)
df_train.drop('Exterior2nd',axis=1, inplace = True)
df_train.drop('Electrical',axis=1, inplace = True)
df_train.drop('GarageType',axis=1, inplace = True)
df_train.drop('RoofStyle',axis=1, inplace = True)
df_train.drop('RoofMatl',axis=1, inplace = True)
df_train.drop('SaleCondition',axis=1, inplace = True)
df_train.drop('Functional', axis=1,inplace = True)


# O mapa de calor mostra as variaveis numericas OverallQual, GrLivArea, ExterQual, 
# KitchenQual, 'GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF e BsmtQual 
# como mais correlacionadas ao preço de venda da casa
for var in cols:
    if var != 'SalePrice':
        data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
# O scatterplot das variáveis mostra que existem alguns outliers nas features mais correlacionadas. 
# Vamos eliminar estes outliers
df_train[:train_len].drop(df_train[
        (df_train['GrLivArea']>4000) & (df_train['SalePrice']<200000)
        ].index, inplace=True)
df_train[:train_len].drop(df_train[
        (df_train['1stFlrSF']>3000) & (df_train['SalePrice']<200000)
        ].index, inplace=True)
df_train[:train_len].drop(df_train[
        (df_train['TotalBsmtSF']>6000) & (df_train['SalePrice']<200000)
        ].index, inplace=True)
df_train[:train_len].drop(df_train[
        (df_train['GarageArea']>1200)
        ].index, inplace=True)
    
# Agora, observemos a distribuição destas variáveis e também do target
sns.distplot(df_train['GrLivArea'] , bins=50,fit=norm);
plt.show()
sns.distplot(df_train['1stFlrSF'] , bins=50,fit=norm);
plt.show()
sns.distplot(df_train['TotalBsmtSF'] , bins=50,fit=norm);
plt.show()
sns.distplot(df_train['GarageArea'] , bins=50,fit=norm);
plt.show()
sns.distplot(df_train[:train_len]['SalePrice'], bins=50, fit=norm);
plt.show()

# Observa-se que algumas variáveis e o alvo possuem assimetria (skewness) acentuada.
# Para normalizar esta situação e facilitar o treinamento dos modelos,
# Aplicaremos transformação logaritmica nas variaveis que apresentarem essa caracteristica.

df_train['SalePrice'][:train_len] = np.log1p(df_train['SalePrice'][:train_len])

skewness = df_train.select_dtypes(exclude='object').apply(lambda x: stats.skew(x))
skewness = skewness[abs(skewness) > 0.6]
skewed_features = skewness.index
df_train[skewed_features] = np.log1p(df_train[skewed_features])
 

    
# Gerar dummies das categoricas
df_train = pd.get_dummies(df_train)

    
# 1o treinamento
from sklearn.linear_model import LinearRegression, LassoCV

train_set = df_train[:train_len]
test_set = df_train[train_len:].drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
        train_set.drop('SalePrice',axis=1), train_set.SalePrice)

sns.distplot(Y_train)
sns.distplot(Y_test)

# Padronização das features para media 0 e variância 1
stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)


scr_tr_lr = lr.score(X_train, Y_train)
scr_ts_lr = lr.score(X_test, Y_test)
print(scr_tr_lr,scr_ts_lr)
pred_train_lr = lr.predict(X_train)
pred_test_lr = lr.predict(X_test)

plt.scatter(pred_train_lr, Y_train, c = "blue", marker = "s", label = "Treino")
plt.scatter(pred_test_lr, Y_test, c = "lightgreen", marker = "s", label = "Validação")
plt.title("Regressão Linear sem Regularização")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# A regressão linear se mostrou inutilizavel neste caso.
# Agora com regularização Lasso, esperamos obter uma performance muito superior.
lass_alphas = [0.00005, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
lass = LassoCV(alphas = lass_alphas, 
               max_iter = 100000,
               cv = 10)
lass.fit(X_train, Y_train)
best_alpha = lass.alpha_
print("Alpha inicial:", best_alpha)
# Agora inicializando a busca pelo melhor alpha a partir dos valores anteriores
new_alphas = [ x * best_alpha for x in lass_alphas ]
lass = LassoCV(alphas = new_alphas, 
               max_iter = 10000,
               cv = 10)
lass.fit(X_train, Y_train)
print("Alpha final:", lass.alpha_)

# 
scr_tr_lass = lass.score(X_train, Y_train)
scr_ts_lass = lass.score(X_test, Y_test)
print(scr_tr_lass,scr_ts_lass)
prediction_test_lass = lass.predict(X_test)
prediction_train_lass = lass.predict(X_train)

# Vamos observar a performance do modelo utilizando Lasso
plt.scatter(prediction_train_lass, Y_train, c = "blue", marker = "s", label = "Treino")
plt.scatter(prediction_test_lass, Y_test, c = "lightgreen", marker = "s", label = "Validação")
plt.title("Regressão Linear com Regularização Lasso")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Gostei do modelo gerado com Lasso, farei a submissão para o Kaggle
# Fazendo a transformação exponencial para compensar a logaritmica anterior
pred = lass.predict(stdSc.transform(test_set))
lass_sub = pd.DataFrame()
lass_sub['Id'] = df_test['Id']
lass_sub['SalePrice'] = np.expm1(pred)
sns.distplot(lass_sub['SalePrice'])
lass_sub.to_csv('submission_lass.csv', index = False)



# O número de features ainda está muito alto... A seguir vou tentar reduzir
# a quantidade de features e observar o comportamento dos modelos.
## Neste ponto, diminuí o numero de features categoricas e criei features artificiais
# Para melhorar a tolerancia a outliers, vou utilizar o RobustScaler ao invés do
# StandartScaler desta vez, também para o lasso.

rbSc = RobustScaler()

train_set = df_train[:train_len]
test_set = df_train[train_len:].drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
        train_set.drop('SalePrice',axis=1), train_set.SalePrice)

X_train = rbSc.fit_transform(X_train)
X_test = rbSc.transform(X_test)

lass_alphas = [0.00005, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
lass = LassoCV(alphas = lass_alphas, 
               max_iter = 100000,
               cv = 10)
lass.fit(X_train, Y_train)
best_alpha = lass.alpha_
print("Alpha inicial:", best_alpha)
# Agora inicializando a busca pelo melhor alpha a partir dos valores anteriores
new_alphas = [ x * best_alpha for x in lass_alphas ]
lass = LassoCV(alphas = new_alphas, 
               max_iter = 100000,
               cv = 10)
lass.fit(X_train, Y_train)
print("Alpha final:", lass.alpha_)
scr_tr_lass = lass.score(X_train, Y_train)
scr_ts_lass = lass.score(X_test, Y_test)
print(scr_tr_lass,scr_ts_lass)
prediction_test_lass = lass.predict(X_test)
prediction_train_lass = lass.predict(X_train)

plt.scatter(prediction_train_lass, Y_train, c = "blue", marker = "s", label = "Treino")
plt.scatter(prediction_test_lass, Y_test, c = "lightgreen", marker = "s", label = "Validação")
plt.title("Lasso e RobustScaler")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

pred = lass.predict(rbSc.transform(test_set))
lass_sub = pd.DataFrame()
lass_sub['Id'] = df_test['Id']
lass_sub['SalePrice'] = np.expm1(pred)
sns.distplot(lass_sub['SalePrice'])
lass_sub.to_csv('submission_lass_rd.csv', index = False)
