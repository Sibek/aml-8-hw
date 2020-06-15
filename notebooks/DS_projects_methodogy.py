#!/usr/bin/env python
# coding: utf-8

# # 1. Понимание бизнеса
# 
# ## 1.1 Цель
# Предсказание оценки качества вина на основе данных из физико-химических свойств
# 
# ## 1.2 Описание
# Рассматривается набор данных с красными и белыми португальскими винами "Vinho Verde". Доступны только физико-химические (входные) и сенсорные (выходные) переменные (например, нет данных о типах винограда, марке вина, цене продажи вина и т. д.).

# # 2. Data Understanding
# 
# ## 2.1 Import Libraries

# In[2]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier

# Modelling Helpers
# from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import  Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
import itertools as it

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# ## 2.2 Вспомогательные функции

# In[3]:


def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 8 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))


# ## 2.3 Загрузка данных

# In[6]:


df = pd.read_csv('/Users/Sibek/Documents/Образование/Data Scientist/5. Машинное обучение/aml-8-hw/data/winequalityN.csv')
df.shape


# ## 2.4 Статистика и визуализации

# In[7]:


df.head()


# **Описание переменных**
# 
# - type - тип вина (красное/белое)
# - fixed acidity - фиксированная кислотность
# - volatile acidity - летучая кислотность
# - citric acid - лимонная кислота
# - residual sugar - остаточный сахар
# - chlorides - хлориды
# - free sulfur dioxide - свободный диоксид серы
# - total sulfur dioxide - общий диоксид серы
# - density - плотность
# - pH
# - sulphates - сульфаты
# - alcohol - алкоголь
# 
# **Output variable (based on sensory data)**
# - quality (score between 0 and 10) - оценка качества

# ### 2.4.1 Ключевая информацию о переменных

# In[8]:


df.describe().transpose()


# В датасете 12 числовых и одна категориальная переменные. Также разу видно, что часть переменных имеет незаполненные значения

# ### 2.4.2 Тепловая карта корреляции может дать нам понимание того, какие переменные важны

# In[23]:


plot_correlation_map(df)
plt.savefig("correlation_map.png")


# Наиболее высокую корреляцию с ключевым показателем имеют показатели: alcohol и density (отрицательную)

# ### 2.4.3 Взаимосвязи между признаками и оценкой качества вина
# Начнем с рассмотрения взаимосвязи алкогольности и оценки качества. При этом переведем показатель качества вина в бинарный вид, принимающий только два значения (0 и 1).

# In[21]:


bins = [0,5,10]

labels = [0, 1]
df['quality_range']= pd.cut(x=df['quality'], bins=bins, labels=labels)

df.head()


# In[24]:


plot_distribution( df , var = 'alcohol', target = 'quality_range', row = 'type' )
plt.savefig("quality_alcohol.png")


# Из полученных графиков можно сделать вывод, что как и у красного вина, так и у белого, его алкогольность заметно влияет на оценку качества. Менее алкогольные вина получают более низкие оценки.

# ### 2.4.4 Тип вина (цвет)
# Мы также можем посмотреть на категориальную переменную type и ее связь с оценкой качества

# In[26]:


sns.countplot(x = 'type', hue = 'quality_range', data = df)
plt.savefig("count_type.png")
plt.show()


# В целом в датасете представлено больше белых вин, однако хорошо заметно, что из всех белых вин около 70% получают оценку 5 и более, в то время как у красных вин только чуть более 50.

# ### 2.4.5 Рассмотрим распределение по основным переменным

# In[27]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df.alcohol)
plt.savefig("alco_distr.png")
plt.show()


# In[28]:


plt.figure(figsize = (10,8))
sns.distplot(df.alcohol, bins=50)
plt.savefig("alco_distr_2.png")
plt.show()


# Медиана по алкогольности по датасету находится на уровне чуть более 10%. В целом алкогольность колеблится от 8% до 14%, есть единичные вина с алкогольностью свыше 14% (выбросы).

# In[29]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df.density)
plt.savefig("density_distr.png")
plt.show()


# In[30]:


plt.figure(figsize = (10,8))
sns.distplot(df.density, bins=300)
plt.savefig("density_distr_2.png")
plt.show()


# Плотность большинства рассматриваемых вин находится между значениями 0.99 и 1. Но есть и выброс со значениями более 1.01.

# # 3. Data Preparation

# ## 3.1 Категориальная переменная должна быть преобразована в числовую переменную

# In[12]:


df.loc[df['type']=='white', 'type'] = 1
df.loc[df['type']=='red', 'type'] = 0
df.head()


# ## 3.2 Заполнить пропущенные значения в переменных
# Большинство алгоритмов машинного обучения требуют, чтобы все переменные имели значения, чтобы использовать их для обучения модели.

# In[13]:


df.isnull().sum()


# Так как пустых значений всего около 30, что составляет менее 1% от всего датасета, предлагается их удалить

# In[14]:


df.dropna(axis=0,inplace=True)


# In[15]:


df.shape


# In[16]:


df.isnull().sum()


# ## 3.3 Сборка финальных датасетов для моделирования

# ### 3.3.1 Создание датасетов
# 
# Отделяем данные для обучения и для проверки

# In[112]:


X = df.drop(['quality', 'quality_range'], axis=1)
y = df[['quality_range']]


# Разделим данные на 80% тренировочных и на 20% тестовых

# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# ### 3.3.2 Важность признаков
# Отбор оптимальных признаков для модели имеет важное значение. Теперь мы попытаемся оценить, какие переменные являются наиболее важными, чтобы сделать прогноз.

# In[114]:


plot_variable_importance(X_train, y_train)


# # 4. Моделирование
# Теперь мы выберем модель, которую хотели бы попробовать. Используем обучающий набор данных для обучения модели и затем проверим ее с помощью тестового набора. А также проведем оценку модели.
# 
# ## 4.1 Выбор, обучение и оценка модели

# In[115]:


columns = X.columns

linear_model = LinearRegression()
forest_model = RandomForestClassifier()
KNeighbors_model = KNeighborsClassifier()
xgb_model = XGBClassifier()
gb_model = GradientBoostingClassifier()
dt_model = DecisionTreeClassifier()
gnb_model = GaussianNB()
svc_model = SVC()

linear_model.fit(X_train[columns], y_train)
forest_model.fit(X_train[columns], y_train)
KNeighbors_model.fit(X_train[columns], y_train)
xgb_model.fit(X_train[columns], y_train)
gb_model.fit(X_train[columns], y_train)
dt_model.fit(X_train[columns], y_train)
gnb_model.fit(X_train[columns], y_train)
svc_model.fit(X_train[columns], y_train)

print('LinearRegression -', round(linear_model.score(X_test[columns], y_test),2))
print('RandomForestClassifier -', round(forest_model.score(X_test[columns], y_test),2))
print('KNeighborsClassifier -', round(KNeighbors_model.score(X_test[columns], y_test),2))
print('XGBClassifier -', round(xgb_model.score(X_test[columns], y_test),2))
print('GradientBoostingClassifier -', round(gb_model.score(X_test[columns], y_test),2))
print('DecisionTreeClassifier -', round(dt_model.score(X_test[columns], y_test),2))
print('GaussianNB -', round(gnb_model.score(X_test[columns], y_test),2))
print('SVC -', round(svc_model.score(X_test[columns], y_test),2))


# Наилучший вариант дает RandomForestClassifier

# # 5. Развертывание

# In[123]:


prediction = forest_model.predict(X_test[columns])
unique_index = X_test.index
test = pd.DataFrame( { 'WineId': unique_index , 'Quality': prediction } )
test.shape
test.head()
# test.to_csv( 'wine_pred.csv' , index = False )

