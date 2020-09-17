
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[173]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[174]:


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[175]:


# def get_list_of_university_towns():
#     '''Returns a DataFrame of towns and the states they are in from the 
#     university_towns.txt list. The format of the DataFrame should be:
#     DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
#     columns=["State", "RegionName"]  )
    
#     The following cleaning needs to be done:

#     1. For "State", removing characters from "[" to the end.
#     2. For "RegionName", when applicable, removing every character from " (" to the end.
#     3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
#     states2 = dict([(value,key) for key, value in states.items()])
#     statesdf = pd.DataFrame.from_dict(states2, orient='index')
#     statesdf= statesdf.reset_index()
#     statesdf = statesdf.rename(columns={'index':'State', 0: 'StateAbb'})
#     city_zhvi_allhomes = pd.read_csv("City_Zhvi_AllHomes.csv")
#     university_towns = pd.read_fwf('university_towns.txt', names=['RegionName'])
#     university_towns = university_towns[(university_towns['RegionName'].str.contains("\[ed")) | (university_towns['RegionName'].str.contains("\("))]
#     #university_towns = university_towns[(university_towns['RegionName'].str.contains("\("))]
#     university_towns['State'] = pd.DataFrame(university_towns['RegionName'].apply(lambda x: x.split('[ed')[0] if type(x)!=np.float else ""))
#     university_towns['RegionName'] = pd.DataFrame(university_towns['RegionName'].apply(lambda x: x.split(' (')[0] if type(x)!=np.float else x))
#     university_towns['RegionName'] = pd.DataFrame(university_towns['RegionName'].apply(lambda x: x.split('\n')[0] if type(x)!=np.float else x))
#     university_towns['State'] = pd.DataFrame(university_towns['State'].apply(lambda x: np.NaN if ' (' in x else x))
#     university_towns['State'] = university_towns['State'].ffill(axis=0)
    
#     university_towns = university_towns[~(university_towns['RegionName'].str.contains("\["))]
#     university_towns=university_towns[['State','RegionName']]
#     return university_towns.shape
# get_list_of_university_towns()
#get_list_of_university_towns()#.groupby('State').size()
#alternate way

def get_list_of_university_towns():
    lst=[]
    state=''
    region=''
    university_town = open('university_towns.txt','r')
    lines = university_town.readlines()
    
    for line in lines:
        if '[ed' in line:
            state = line.split('[')[0]
        else:
            region = line.split(' (')[0]
            
            lst.append([state,region])
            
    university_town = pd.DataFrame(lst)
    university_town = university_town.rename(columns={0:'State', 1:'RegionName'})
    university_town['RegionName'] = pd.DataFrame(university_town['RegionName'].apply(lambda x: x.split('\n')[0] if type(x)!=np.float else x))
            
    return university_town

#get_list_of_university_towns()#.groupby('State').size()


# In[176]:


# import re
# import pandas as pd
# import numpy as np
# # list of unique states
# stateStr = """
# Ohio, Kentucky, American Samoa, Nevada, Wyoming
# ,National, Alabama, Maryland, Alaska, Utah
# ,Oregon, Montana, Illinois, Tennessee, District of Columbia
# ,Vermont, Idaho, Arkansas, Maine, Washington
# ,Hawaii, Wisconsin, Michigan, Indiana, New Jersey
# ,Arizona, Guam, Mississippi, Puerto Rico, North Carolina
# ,Texas, South Dakota, Northern Mariana Islands, Iowa, Missouri
# ,Connecticut, West Virginia, South Carolina, Louisiana, Kansas
# ,New York, Nebraska, Oklahoma, Florida, California
# ,Colorado, Pennsylvania, Delaware, New Mexico, Rhode Island
# ,Minnesota, Virgin Islands, New Hampshire, Massachusetts, Georgia
# ,North Dakota, Virginia
# """
# #list of regionName entries string length
# regNmLenStr = """
# 06,08,12,10,10,04,10,08,09,09,05,06,11,06,12,09,08,10,12,06,
# 06,06,08,05,09,06,05,06,10,28,06,06,09,06,08,09,10,35,09,15,
# 13,10,07,21,08,07,07,07,12,06,14,07,08,16,09,10,11,09,10,06,
# 11,05,06,09,10,12,06,06,11,07,08,13,07,11,05,06,06,07,10,08,
# 11,08,13,12,06,04,08,10,08,07,12,05,06,09,07,10,16,10,06,12,
# 08,07,06,06,06,11,14,11,07,06,06,12,08,10,11,06,10,14,04,11,
# 18,07,07,08,09,06,13,11,12,10,07,12,07,04,08,09,09,13,08,10,
# 16,09,10,08,06,08,12,07,11,09,07,09,06,12,06,09,07,10,09,10,
# 09,06,15,05,10,09,11,12,10,10,09,13,06,09,11,06,11,09,13,37,
# 06,13,06,09,49,07,11,12,09,11,11,07,12,10,06,06,09,04,09,15,
# 10,12,05,09,08,09,09,07,14,06,07,16,12,09,07,09,06,32,07,08,
# 08,06,10,36,09,10,09,06,09,11,09,06,10,07,14,08,07,06,10,09,
# 05,11,07,06,08,07,05,07,07,04,06,05,09,04,25,06,07,08,05,08,
# 06,05,11,09,07,07,06,13,09,05,16,05,10,09,08,11,06,06,06,10,
# 09,07,06,07,10,05,08,07,06,08,06,30,09,07,06,11,07,12,08,09,
# 16,12,11,08,06,04,10,10,15,05,11,11,09,08,06,04,10,10,07,09,
# 11,08,26,07,13,05,11,03,08,07,06,05,08,13,10,08,08,08,07,07,
# 09,05,04,11,11,07,06,10,11,03,04,06,06,08,08,06,10,09,05,11,
# 07,09,06,12,13,09,10,11,08,07,07,08,09,10,08,10,08,56,07,12,
# 07,16,08,04,10,10,10,10,07,09,08,09,09,10,07,09,09,09,12,14,
# 10,29,19,07,11,12,13,13,09,10,12,12,12,08,10,07,10,07,07,08,
# 08,08,09,10,09,11,09,07,09,10,11,11,10,09,09,12,09,06,08,07,
# 12,09,07,07,06,06,08,06,15,08,06,06,10,10,10,07,05,10,07,11,
# 09,12,10,12,04,10,05,05,04,14,07,10,09,07,11,10,10,10,11,15,
# 09,14,12,09,09,07,12,04,10,10,06,10,07,28,06,10,08,09,10,10,
# 10,13,12,08,10,09,09,07,09,09,07,11,11,13,08,10,07
# """

# df = get_list_of_university_towns()

# cols = ["State", "RegionName"]

# print('Shape test: ', "Passed" if df.shape ==
#       (517, 2) else 'Failed')
# print('Index test: ',
#       "Passed" if df.index.tolist() == list(range(517))
#       else 'Failed')

# print('Column test: ',
#       "Passed" if df.columns.tolist() == cols else 'Failed')
# print('\\n test: ',
#       "Failed" if any(df[cols[0]].str.contains(
#           '\n')) or any(df[cols[1]].str.contains('\n'))
#       else 'Passed')
# print('Trailing whitespace test:',
#       "Failed" if any(df[cols[0]].str.contains(
#           '\s+$')) or any(df[cols[1]].str.contains(
#               '\s+$'))
#       else 'Passed')
# print('"(" test:',
#       "Failed" if any(df[cols[0]].str.contains(
#           '\(')) or any(df[cols[1]].str.contains(
#               '\('))
#       else 'Passed')
# print('"[" test:',
#       "Failed" if any(df[cols[0]].str.contains(
#           '\[')) or any(df[cols[1]].str.contains(
#               '\]'))
#       else 'Passed')

# states_vlist = [st.strip() for st in stateStr.split(',')]

# mismatchedStates = df[~df['State'].isin(
#     states_vlist)].loc[:, 'State'].unique()
# print('State test: ', "Passed" if len(
#     mismatchedStates) == 0 else "Failed")
# if len(mismatchedStates) > 0:
#     print()
#     print('The following states failed the equality test:')
#     print()
#     print('\n'.join(mismatchedStates))

# df['expected_length'] = [int(s.strip())
#                          for s in regNmLenStr.split(',')
#                          if s.strip().isdigit()]
# regDiff = df[df['RegionName'].str.len() != df['expected_length']].loc[
#     :, ['RegionName', 'expected_length']]
# regDiff['actual_length'] = regDiff['RegionName'].str.len()
# print('RegionName test: ', "Passed" if len(regDiff) ==
#       0 else ' \nMismatching regionNames\n {}'.format(regDiff))


# In[177]:


# def get_list_of_university_towns():
#     '''Returns a DataFrame of towns and the states they are in from the 
#     university_towns.txt list. The format of the DataFrame should be:
#     DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
#     columns=["State", "RegionName"]  )
    
#     The following cleaning needs to be done:

#     1. For "State", removing characters from "[" to the end.
#     2. For "RegionName", when applicable, removing every character from " (" to the end.
#     3. Depending on how you read the data, you may need to remove newline character '\n'. '''
#     states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

#     states2 = dict([(value, key)for key, value in states.items()])
#     statesdf = pd.DataFrame.from_dict(states2, orient='index')
#     university_towns = pd.read_fwf('university_towns.txt', names=['RegionName'])
#     university_towns1 = university_towns.copy()
#     city_zhvi_allhomes = pd.read_csv('City_Zhvi_AllHomes.csv')
#     city_zhvi_allhomes2 = city_zhvi_allhomes.copy()
#     city_zhvi_allhomes2 = city_zhvi_allhomes2.sort_values(['State'])
#     university_towns = pd.DataFrame(university_towns['RegionName'].apply(lambda x: x.split('[')[0] if (type(x)!= np.float) else x))
#     university_towns = pd.DataFrame(university_towns['RegionName'].apply(lambda x: x.split(' (')[0] if (type(x)!= np.float) else x))
#     university_towns['StateAbb'] = university_towns['RegionName'].map(states2)
#     university_towns['State'] =  university_towns['StateAbb'].map(states)
#     city_zhvi_allhomes.drop(city_zhvi_allhomes.columns.difference(['RegionName','State']), 1, inplace=True)
#     city_zhvi_allhomes1 = city_zhvi_allhomes.copy()
#     city_zhvi_allhomes1 = city_zhvi_allhomes1.sort_values(['State'])
    
#     #
#     university_towns =  university_towns.ffill(axis=0)
#     university_towns = university_towns.sort_values(['State'])

#     university_towns =  university_towns[~(university_towns['RegionName'].isin(states2).values)]
    
#     university_towns2 = university_towns.copy()
#     #university_towns2 = university_towns.groupby(['State']).agg(['count'])
   
#     university_towns =  university_towns.set_index(['StateAbb','State']).sort_index()
    
#     city_zhvi_allhomes = city_zhvi_allhomes.rename(columns={'State':'StateAbb'})
#     city_zhvi_allhomes['State'] = city_zhvi_allhomes['StateAbb'].map(states)
#     city_zhvi_allhomes = city_zhvi_allhomes.set_index(['StateAbb','State']).sort_index()
#     #city_zhvi_allhomes = city_zhvi_allhomes
#     #city_zhvi_allhomes =  city_zhvi_allhomes[(city_zhvi_allhomes['RegionName'].isin(university_towns['RegionName']).values)]
#     newdf =  university_towns.merge(city_zhvi_allhomes, how='outer', left_index=True, right_index=True)
#     #newdf = newdf.drop_duplicates(subset=['RegionName_x','RegionName_y'],keep='first')
#     #newdf = newdf.reset_index()
#     #newdf = newdf.dropna()
#     #newdf.drop(newdf.columns.difference(['RegionName','State']),1,inplace=True)
#     #newdf = newdf[['State','RegionName']]
#     #newdf = newdf.sort_values(['State'])
#     #newdf =  newdf[~(newdf['RegionName'].isin(states2).values)]
#     #newdf = city_zhvi_allhomes1
#     return  newdf
# get_list_of_university_towns()


# In[178]:


def get_recession_start():
# A quarter is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    
    gdplev = pd.read_excel('gdplev.xls', skiprows=5, header=0)
    #annual = gdplev.iloc[2:, 0:3]
    #annual = annual.rename(columns={'Unnamed: 0':'Year'})
    #annual = annual.dropna()
    #annual['Year'] = annual['Year'].astype(int)
    quarterly = gdplev.iloc[:, 4:7]
    quarterly = quarterly.dropna()
    quarterly = quarterly.rename(columns={'Unnamed: 4': 'Quarter'})
    quarterly['Year'] = quarterly['Quarter'].apply(lambda x: x.split('q')[0]).astype(int)
    #quarterly['Year'] = quarterly['Year'].astype(int)
    
    #more than 2000 years
    quarterly = quarterly[quarterly['Year']>=2000]
    #annual = annual[annual['Year']>=2000]
    #quarterly = quarterly.set_index('Year')
    #annual = annual.set_index('Year')
    #newdf = quarterly.merge(annual, how='outer', right_index=True, left_index=True)
    #quarterly = quarterly[['Year','Quarter','GDP in billions of current dollars.1', 'GDP in billions of chained 2009 dollars.1']]
    #pseudocode
    quarterly = quarterly.set_index(['Year','Quarter'])
    quarterly['Next'] = quarterly['GDP in billions of current dollars.1'].shift(-1)
    quarterly['Diff'] = quarterly['Next'] < quarterly['GDP in billions of current dollars.1']
    quarterly = quarterly.reset_index()
    #quarters = quarterly['Quarter'].tolist()
    zip_diff = list(zip(quarterly['Diff'], quarterly['Diff'][1:]))
    #newdf = quarterly.Diff.isin(boolean_values)
    index_no = zip_diff.index((True,True))
    #quarterly.loc[(34*2)]
  
        
    #newdf = quarterly.set_index('Diff').loc[boolean_values]
    #newdf = quarterly[quarterly['Diff']==boolean_values]
    #gdplev=gdplev.drop(['Unnamed: 3'], axis=1)
    #gdplev = gdplev.rename(columns={'Unnamed: 0':'Year','Unnamed: 4': 'Quarter', 'Unnamed: 3':'YearLabel'})
   # gdplev['Year'] = pd.to_numeric(gdplev['Year'], errors='coerce')
    #gdplev['Year'] = gdplev['Year'].astype(int)
    #df['x'] = df['x'].astype(int)
    #years = gdplev[gdplev['Year'].dropna].tolist()
    #years = pd.DataFrame(gdplev['Year'].dropna())
    #years = pd.DataFrame(years.astype(int))#
    #years = years[years['Year']>=2000]['Year'].tolist()
    
    #gdplev = gdplev[gdplev['Quarter'].str.startswith("20")]
    #gdplev['YearLabel'] = gdplev['Quarter'].apply(lambda x: x.split('q')[0]).astype(int)
    
    #gdplev['Year'] = gdplev['Year'].apply(lambda x: int(x) if (type(x) == np.float & (~x==np.nan)) else x)
    #gdplev['Quarter'] = gdplev['Quarter'].apply(lambda x: )
    #df.drop(columns=['B', 'C'])
    #pseudocode
    #startswith 20
    #if gdp decline consecutive two times, and ending w/ 2 consecutive of gdp growth
    return quarterly.iloc[index_no]['Quarter']
#get_recession_start()


# In[179]:


def get_recession_end():
    recession_start = get_recession_start()
    
    gdplev = pd.read_excel('gdplev.xls', skiprows=5, header=0)
    
    quarterly = gdplev.iloc[:, 4:7]
    quarterly = quarterly.dropna()
    quarterly = quarterly.rename(columns={'Unnamed: 4': 'Quarter'})
    quarterly['Year'] = quarterly['Quarter'].apply(lambda x: x.split('q')[0]).astype(int)

    quarterly = quarterly[quarterly['Year']>=2000]
    
    quarterly = quarterly.set_index(['Year','Quarter'])
    quarterly['Next'] = quarterly['GDP in billions of current dollars.1'].shift(-1)
    quarterly['Diff'] = quarterly['Next'] > quarterly['GDP in billions of current dollars.1']
    quarterly = quarterly.reset_index()
    after_row = (quarterly.loc[quarterly['Quarter']==recession_start].index.values.item())+2
    #quarterly = quarterly.iloc[after_row:]
    zip_diff = list(zip(quarterly['Diff'], quarterly['Diff'][1:], quarterly['Diff'][2:], quarterly['Diff'][3:]))
    
    #after_row = quarterly.index[quarterly['Quarter']>=recession_start].tolist()
    index_no = (zip_diff.index((False, False, True,True)))+4
    
    #del zip_diff[0:after_row]
    return quarterly.iloc[index_no]['Quarter']
#get_recession_end()


# In[180]:


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    start_recession = get_recession_start()
    end_recession = get_recession_end()
    
    #A recession bottom is the quarter within a recession which had the lowest GDP.
    gdplev = pd.read_excel('gdplev.xls', skiprows=5, header=0)
    quarterly = gdplev.iloc[:, 4:7]
    quarterly = quarterly.dropna()
    quarterly = quarterly.rename(columns={'Unnamed: 4': 'Quarter'})
    quarterly['Year'] = quarterly['Quarter'].apply(lambda x: x.split('q')[0]).astype(int)
    
    quarterly = quarterly[quarterly['Year']>=2000]
    start_index = quarterly.loc[quarterly['Quarter']==start_recession].index.values.item()
    end_index = quarterly.loc[quarterly['Quarter']==end_recession].index.values.item()
    quarterly = quarterly.loc[start_index:end_index]
    quarterly = quarterly.set_index('Quarter')
    
    lowest = quarterly['GDP in billions of current dollars.1'].min()
    #lowest_quarter = quarterly['GDP in billions of current dollars.1']==lowest
    return quarterly.loc[quarterly['GDP in billions of current dollars.1']==lowest].index.values.item()


# In[181]:


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    'A quarter is a specific three month period,'
    'Q1 is January through March, Q2 is April through June, '
    'Q3 is July through September, Q4 is October through December.'
    
    datelist = []
    
    for i in range(0, 17, 1):
        new_no = 2000 + i
        datelist.append(new_no)
        
    city_zhvi_allhomes = pd.read_csv("City_Zhvi_AllHomes.csv")
    #city_zhvi_allhomes = city_zhvi_allhomes.dropna()
    city_zhvi_allhomes=city_zhvi_allhomes.drop(['RegionID','Metro','CountyName','SizeRank'], axis=1)
    datecolumns = city_zhvi_allhomes.iloc[0:, 6:]
    indexcolumns= city_zhvi_allhomes.iloc[0:, 0:2]
    datecolumns.columns = pd.to_datetime(datecolumns.columns)
    
    date = datecolumns.columns.get_loc('2000-01-01 00:00:00')
    
    datecolumns = datecolumns.iloc[:, date:]
    citycol = datecolumns.merge(indexcolumns, how='outer', left_index=True, right_index=True)
    
    newdf = pd.DataFrame(np.add.reduceat(datecolumns.values, np.arange(len(datecolumns.columns))[::3], axis=1))
    newdf = newdf/3
    #citycol = pd.DataFrame(np.add.reduceat(citycol.values, np.arange(len(citycol.columns))[::3], axis=1))
#     citycol = citycol.merge(indexcolumns, how='outer', left_index=True, right_index=True)
#     citycol = citycol.set_index(['State','RegionName'])
#     citycol = citycol.drop([50], axis=1)
    columnnames = []
    for i in range(0, 17, 1):
        new_no = 2000 + i
        
        for y in range (1,5,1):
            new_no1 = y
            new_name = str(new_no) +"q"+ str(new_no1)

            columnnames.append(new_name)
    
    columnnames.pop()
    
    newdf.columns = columnnames
    #citycol.columns=columnnames
    #citycol=citycol.shift(periods=1,axis='rows')
    #citycol = citycol.dropna()
    #citycol = citycol[1:]
    #indexcolumns = indexcolumns.dropna()
    #newdf = newdf.dropna()
    newdf = newdf.merge(indexcolumns, how='outer', left_index=True, right_index=True)
    newdf['State']=newdf['State'].replace(states)
    newdf = newdf.set_index(['State','RegionName'])
    #newdf=newdf.shift(periods=1,axis='rows')[1:]
    #newdf=newdf.mean(axis=0)
    #newdf=newdf[1:]
    #newdf = newdf.dropna()
    #print(indexcolumns)
    
    return newdf
#convert_housing_data_to_quarters().loc["Texas"].loc["Austin"].loc["2010q3"]


# In[182]:


from scipy import stats


# In[272]:


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    alternative hypothesis = two groups are the same.
    null hypothesis = two groups are not the same. 
    
    #t-test true, different=true, which means null hypothesis is both are the same
    alternate hypothesis is both are different
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    'Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession '
    'starts compared to the recession bottom. (price_ratio=quarter_before_recession/recession_bottom)'
    
    house_data = convert_housing_data_to_quarters().reset_index()
    house_data_2 = house_data.copy()
    uni_towns = get_list_of_university_towns()
    house_data_2 = house_data_2.set_index(['State','RegionName'])
    uni_towns['is_uni_town'] = True
    
    uni_town_values = house_data[((house_data['State'].isin(uni_towns['State'])) & (house_data['RegionName'].isin(uni_towns['RegionName'])))]
    non_uni_town_values = house_data[~((house_data['State'].isin(uni_towns['State'])) & (house_data['RegionName'].isin(uni_towns['RegionName'])))]
    uni_towns = uni_towns.set_index(['State','RegionName'])
    newdf = house_data_2.merge(uni_towns, how='outer',left_index=True,right_index=True)
    newdf['is_uni_town'] = newdf['is_uni_town'].fillna(False)
    new_uni_towns = newdf[newdf['is_uni_town']==True]
    new_non_uni_towns = newdf[newdf['is_uni_town']==False]
    
    start_recession = get_recession_start()
    start_bottom = get_recession_bottom()

    before_recession = uni_town_values.columns[uni_town_values.columns.get_loc(start_recession)]
    
    new_uni_towns['price_ratio'] = new_uni_towns[start_recession]/new_uni_towns[start_bottom]
    new_non_uni_towns['price_ratio'] = new_non_uni_towns[start_recession]/new_non_uni_towns[start_bottom]
    new_uni_towns = new_uni_towns.dropna()
    new_non_uni_towns = new_non_uni_towns.dropna()
    print(new_uni_towns.size)
    print(new_non_uni_towns.size)
    t,p = stats.ttest_ind(new_uni_towns['price_ratio'],new_non_uni_towns['price_ratio'], nan_policy='omit')


    different = p < 0.01
    
    better=''
    if new_uni_towns['price_ratio'].mean()<new_non_uni_towns['price_ratio'].mean():
        better = 'university town'
    else:
        better = 'non-university town'



    return (different,p,better)
run_ttest()


# In[ ]:




