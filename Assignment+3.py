
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[2]:


import pandas as pd
import numpy as np
import re


#GDP[(GDP['Data Source']=='South Korea')]
#for x in energy['Country']:                 
   # if (type(x)==np.str):
        #x = x.split("(",1)[0]
        
#energy['Country'] = energy['Country'].apply(lambda x: x.split(" (")[0] if (type(x)!=np.float) else x)
#energy['Country'] = energy['Country'].apply(lambda x: x.split('(\d+)')[0] if (type(x)==np.string_ && ('(\d+)')) else x)
#energy[energy['Country']=='Switzerland']

def answer_one():
    energy = pd.read_excel('Energy Indicators.xls').drop(0)
    energy=energy.iloc[:-38]
    energy = energy.drop(["Unnamed: 0", "Unnamed: 1"], axis=1)
    energy = energy.rename(columns={"Environmental Indicators: Energy": "Country", "Unnamed: 3": "Energy Supply", "Unnamed: 4": "Energy Supply per Capita", "Unnamed: 5": "% Renewable"})
    
    energy = energy.replace(to_replace=['...'], value=np.nan)
    energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x: x*1000000 if (type(x)!=np.nan) else x)
    energy['Country'] = energy['Country'].astype('str')
    replace_dict = {"Republic of Korea": "South Korea",
"United States of America": "United States",
"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
"China, Hong Kong Special Administrative Region": "Hong Kong"}
    # energy['Country'] = energy['Country'].replace({'Republic of Korea':"South Korea"}).astype('str')
    #energy['Country'] = energy['Country'].replace({'United States of America': "United States"}).astype('str')
    #energy['Country'].replace(to_replace=replace_dict, value=new_value)
    
    #energy = energy.replace({'Country':{'United States of America':'United States','United Kingdom of Great Britain and Northern Ireland':'United Kingdom','China, Hong Kong Special Administrative Region':'Hong Kong','Republic of Korea':'South Korea'}})
    #energy['Country'] = energy.replace({'Country':{:, : ,"United Kingdom of Great Britain and Northern Ireland":'United Kingdom','China, Hong Kong Special Administrative Region':'Hong Kong'}})
    energy['Country'] = energy['Country'].apply(lambda x: x.split(" (")[0] if (type(x)!=np.float) else x)
    pattern='[0-9]'
    energy['Country'] = energy['Country'].apply(lambda x: re.sub(pattern, '', x) if (type(x)!=np.float) else x)
    energy = energy.replace({'Country':replace_dict})
    #print(energy)
   # energy = energy.iloc[8:-37]

    GDP = pd.read_csv('world_bank.csv')
    header = GDP.loc[3]
    GDP = GDP[4:]
    GDP.columns = header
    GDP = GDP.rename(columns={'Country Name':'Country'})
    GDP['Country'] = GDP['Country'].replace({'Korea, Rep.': "South Korea", "Iran, Islamic Rep.": 'Iran', 'Hong Kong SAR, China': "Hong Kong"})
    #
    
    
    list_x = []
    list_y = []
    for x in GDP.columns:
        if type(x)==np.float64:
            list_x.append(x)
            list_y.append('{:0.0f}'.format(x))

    GDP.rename(columns={x:y for x, y in zip(list_x, list_y)}, inplace=True)
    GDP = GDP[['Country', 'Country Code', 'Indicator Name', 'Indicator Code', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015' ]]        

    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    ScimEn = ScimEn.sort_values('Rank').head(15)
    top15 = pd.DataFrame(ScimEn.values)
    top15 = top15.set_index(1)
    top15names = top15.index.tolist()
    GDP = GDP[GDP['Country'].isin(top15names)==True]
    GDP = GDP.set_index('Country')
    energy = energy[energy['Country'].isin(top15names)==True]
    energy = energy.set_index('Country')
    ScimEn = ScimEn.set_index('Country')

    #ScimEn
    #GDP.head(10)
    
    mergedDF = pd.merge(pd.merge(energy, ScimEn, how='outer',left_index=True, right_index=True), GDP,how='outer', left_index=True, right_index=True)
    #mergedDF.rename(columns={"Country_y":"Country"}, inplace=True)
    #mergedDF = mergedDF.sort_values('Rank').head(15)
    #mergedDF = mergedDF.set_index('Country')
    #mergedDF.reset_index()
    dontdrop = ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    #mergedDF = mergedDF.set_index('Country')
    mergedDF.drop(mergedDF.columns.difference(dontdrop), 1, inplace=True)
    mergedDF = mergedDF.sort_values('Rank').head(15)
    #mergedDF = mergedDF[mergedDF['Country'].isin(top15names)]
    #mergedDF.reset_index()
    #mergedDF = mergedDF.set_index('Country_y')
    return mergedDF


# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[3]:


get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[72]:


def answer_two():
    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    ScimEn = ScimEn.set_index(['Country'])
    energy = pd.read_excel('Energy Indicators.xls').drop(0)
    energy = energy.iloc[10:242]#15:242]
    energy = energy.rename(columns={'Unnamed: 1': 'Country'})
   
    GDP = pd.read_csv('world_bank.csv')
    header = GDP.loc[3]
    GDP = GDP[4:]
    GDP.columns = header
    GDP = GDP.rename(columns={'Country Name':'Country'})
    GDP['Country'] = GDP['Country'].replace({'Korea, Rep.': "South Korea", "Iran, Islamic Rep.": 'Iran', 'Hong Kong SAR, China': "Hong Kong"})
    #
    
    replace_dict = {"Republic of Korea": "South Korea", "United States of America": "United States", "United Kingdom of Great Britain and Northern Ireland": "United Kingdom", "China, Hong Kong Special Administrative Region": "Hong Kong"}
    energy['Country'] = energy['Country'].apply(lambda x: x.split(" (")[0] if (type(x)!=np.float) else x)
    pattern='[0-9]'
    energy['Country'] = energy['Country'].apply(lambda x: re.sub(pattern, '', x) if (type(x)!=np.float) else x)
    energy = energy.replace({'Country':replace_dict})
    energy = energy.set_index(['Country'])
    list_x = []
    list_y = []
    for x in GDP.columns:
        if type(x)==np.float64:
            list_x.append(x)
            list_y.append('{:0.0f}'.format(x))

    GDP.rename(columns={x:y for x, y in zip(list_x, list_y)}, inplace=True)
    GDP = GDP[['Country', 'Country Code', 'Indicator Name', 'Indicator Code', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015' ]]        
    
    GDP['Country'] = GDP['Country'].apply(lambda x: x.split(" (")[0] if (type(x)!=np.float) else x)
    pattern='[0-9]'
    GDP['Country'] = GDP['Country'].apply(lambda x: re.sub(pattern, '', x) if (type(x)!=np.float) else x)
    GDP = GDP.replace({'Country':replace_dict})
    GDP = GDP.set_index('Country')
    mergeddf = energy.merge(ScimEn.merge(GDP, how='outer', left_index=True, right_index=True), how='outer', left_index=True, right_index=True)
    r, c = mergeddf.shape
    return r-15


# ## Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[5]:


def answer_three():
    Top15 = answer_one()
    Top15 = Top15.reset_index()
    data = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    #Top15.drop(Top15.columns.difference(data),1,inplace=True)
    avgcalc = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1)
    #colls = ['Country', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    avgGDP = Top15
    avgGDP
    avgGDP.drop(avgGDP.columns.difference(['Country','average']), 1,inplace=True)
    #avgGDP = avgGDP.set_index('Country')
    
    avgGDP['average'] = avgcalc
    avgGDP = pd.Series(avgGDP['average'].values, index=avgGDP['Country'])
    return avgGDP


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[6]:


def answer_four():
    Top15 = answer_one()
    avgGDP = answer_three().to_frame()
    #avgGDP = pd.DataFrame({'Country':avgGDP.index, 'average':avgGDP.values})
    sixthlargest = pd.DataFrame({'average':avgGDP[0]})
    sixthlargest = sixthlargest['average'].nlargest(6)[-1:]
    #sixthlargest = sixthlargest[-1:]
    indexname = sixthlargest.index.values.tolist()
    countrydiff = Top15.loc[indexname]
    countrydiff = countrydiff[['2006', '2015']]
    countrydiff = countrydiff['2015']-countrydiff['2006']
    #countrynamesmall.drop(countrynamesmall.columns.difference(['2006']))
    return countrydiff.iloc[0]
answer_four()


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[7]:


def answer_five():
    Top15 = answer_one()
    mean_energy_supply = Top15['Energy Supply per Capita'].mean()
    return float(mean_energy_supply)


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[8]:


def answer_six():
    Top15 = answer_one()
    max_renewable = Top15['% Renewable'].max()
    country = Top15[(Top15['% Renewable']==max_renewable)].index.tolist()
    return tuple([''.join(country),max_renewable])

#answer_six()


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[9]:


def answer_seven():
    Top15 = answer_one()
    Top15['ratio'] = Top15['Self-citations']/Top15['Citations']
    ratiolargest = Top15['ratio'].nlargest(1)
    #country = Top15.index[(Top15['ratio']==ratiolargest)].tolist()
    #country = ratiolargest[ratiolargest]
    ratiolargest = ratiolargest.reset_index()
    return tuple(ratiolargest.iloc[0])


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[10]:


def answer_eight():
    Top15 = answer_one()
    Top15['population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15 = Top15.sort_values(['population'], ascending = (False))
    indexname = Top15.index.tolist()[2]
    return indexname


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[11]:


def answer_nine():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    #Top15['capita'] = Top15['Energy Supply']/Top15['Energy Supply per Capita']
    #Top15['Citable docs per Capita'] = Top15['Citable documents']/Top15['capita']
    Top15 = Top15[['Energy Supply per Capita', 'Citable docs per Capita']]
    #x = pd.Series(Top15['Energy Supply per Capita'])
    #y = pd.Series(Top15['Citable docs per Capita'])
    #corr_matrix = Top15['Energy Supply per Capita'].astype('float64').corr(Top15['Citable docs per Capita'], method='pearson', min_periods=1).astype('float64')
   # Top15['Energy Supply per Capita']=Top15['Energy Supply per Capita'].astype('float64')
    
    return Top15['Energy Supply per Capita'].astype(float).corr(Top15['Citable docs per Capita'].astype(float), method='pearson')


# In[12]:


def plot9():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[13]:


#plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[14]:


def answer_ten():
    Top15 = answer_one()
    median = np.median(Top15['% Renewable'])
    #median_list = 
    Top15['above_median'] = Top15['% Renewable'].apply(lambda x: 1 if (x>=median) else (0 if (x < median) else x)).astype(int)
    
    HighRenew = pd.Series(Top15['above_median'])
    return HighRenew


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[15]:


def answer_eleven():
    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}
    #data = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    #mergedDF.drop(mergedDF.columns.difference(dontdrop), 1, inplace=True)
    Top15 = answer_one()
    Continent = Top15.copy()
    population = Top15.copy()
    population['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
    #Top15 = Top15.reset_index()
    
    population.drop(Top15.columns.difference(['PopEst']),1,inplace=True)
    ##population['std'] = np.std(population['PopEst'])
    #population['sum'] = population.groupby(by=ContinentDict)['PopEst'].transform(np.sum)
    Continent['size'] = 0
    #population['mean'] = population.groupby(by=ContinentDict)['PopEst'].transform(np.mean)
    #population = population.reset_index()
    #population['Continent'] = population['Country'].map(ContinentDict)
    #population = population.set_index('Continent').groupby(level=0)
    population = population.groupby(by=ContinentDict).agg(['sum', 'mean', 'std'])
    population.columns = population.columns.droplevel()
    #population.drop(population.columns.difference(['std', 'sum', 'mean']), 1, inplace=True)
    #population = population.groupby(by=ContinentDict)
    Continent.drop(Continent.columns.difference(['size']), 1, inplace=True)
    Continent = Continent.groupby(by=ContinentDict).count()
    #Top15.drop(Top15.columns.difference(['Country']),1, inplace=True)
    #Top15['PopEst'] = pop_est
    #Top15 = Top15.groupby(by=ContinentDict).count()#.reset_index(name='count')#.agg(['size','sum', 'mean', 'std'])
    Continent = pd.merge(Continent, population, how='outer', left_index=True, right_index=True)
    #Top15 = Top15.reset_index()
    
    #continentlist=[]
    #for x,y in zip(ContinentDict, Top15['Country']):
        #if(x == y):
            #continentlist.append(ContinentDict[x])
    #Continent = Top15.copy()
    #number of countries in each continent
    
    
    
    #Top15['Continent'] = continentlist
    #Top15 = Top15.drop(Top15.columns.difference(['Continent','PopEst']),1, inplace=True)
    #Continent = Continent.groupby(['Continent']).count().sort_values(['Country']).rename(columns={'Country':'size'})
    #countList = Continent['Continent'].groupby((Continent['Continent']).agg{'continent':'count', }
    #Top15=Top15.groupby('Continent').groupby(level=0).count()
    #Continent = Continent.groupby(counts=['Continent'].count(); 
    #Top15['size'] = Top15['Continent'].count()
    
    
    
        #if (x==y):
            #continentlist.append()
    #Continent = Top15.groupby('')[Top15.index].agg([np.size, np.sum, np.mean])
    #Top15 = Top15.set_index('Country')
    #newdata = Top15.set_index('Country').groupby(level=0)[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].agg([np.sum, np.average])
    #mean = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean()
    #totalsum = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].sum()
    #sd = pd.DataFrame(Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]).std(axis=1, skipna=False)
    #newdata = Top15.copy()
    #.agg([np.sum, np.average, np.std])
    #newdata['sum'] = totalsum
    #newdata['mean'] = mean
    #newdata['std'] = sd
    return Continent
             #.groupby(by=ContinentDict).agg(['size','sum', 'mean', 'std'])


#for x in ContinentDict:
   #print(ContinentDict[x])
    


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[16]:


def answer_twelve():
    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}
    #convert dictionary to tuples
    
    #ContinentTuples = [(k, v) for k, v in ContinentDict.items()]
    Top15 = answer_one()
    Top15 = Top15.reset_index()
    Top15['Continent'] = Top15['Country'].map(ContinentDict)
    
    Top15.drop(Top15.columns.difference(['Continent', 'Country','% Renewable']), 1, inplace=True)
    #Top15=Top15.set_index(['Continent', 'Country'], inplace=False)
    renewable = Top15.copy()
    Top15['% Renewable'] = pd.cut(Top15['% Renewable'],  bins=5, labels=False)
    #renewable = renewable.set_index(['Continent', 'Country'])
    #countries = Top15.copy()
    #countries=countries.set_index(['Continent', 'Country'], inplace=True)
    Top15= Top15.groupby(['Continent', '% Renewable']).count()
    #new_df = Top15.merge(renewable, how='outer', right_index=True, left_index=True)
    
    #pd.MultiIndex.from_tuples(ContinentTuples, names=('Country', 'Continent'))
    #renewable_bin = 
    #renewable_bin = renewable_bin.cut(, bins=5, labels=False)
    #|indexnames = countries[countries['% Renewable']==0].index
    #countries.drop(indexnames,inplace=True)
    return Top15.ix[:,0]#pd.cut(Top15['% Renewable'],  bins=5, labels=False


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[17]:


def answer_thirteen():
    Top15 = answer_one()
    PopEst = Top15.copy()
    PopEst['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
    PopEst['PopEst'] = PopEst['PopEst'].map('{:,}'.format)
    PopEst.drop(PopEst.columns.difference(['PopEst']), 1, inplace=True)
    return pd.Series(PopEst['PopEst'])


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[53]:


def plot_optional():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")


# In[54]:


#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!

