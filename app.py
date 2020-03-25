#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:10:09 2020

@author: diwaker
"""

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import webbrowser

from functools import reduce
from scipy.integrate import odeint
from itertools import chain, repeat, islice
import math


import getPopulation



#Some book keeping, will be needed later
def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

st.title('Corona Lockdown')



#Downloading data from JHU repository
@st.cache
def downloadData():
    df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    return df_confirmed, df_deaths, df_recovered

df_confirmed, df_deaths, df_recovered=downloadData()


# 1.2 Tidying the data
id_list = df_confirmed.columns.to_list()[:4]
vars_list = df_confirmed.columns.to_list()[4:]

confirmed_tidy = pd.melt(df_confirmed, id_vars=id_list,     value_vars=vars_list, var_name='Date', value_name='Confirmed')
deaths_tidy = pd.melt(df_deaths, id_vars=id_list,     value_vars=vars_list, var_name='Date', value_name='Deaths')
recovered_tidy = pd.melt(df_recovered, id_vars=id_list,     value_vars=vars_list, var_name='Date', value_name='Recovered')


# In[5]:


# 1.3 Merging the three dataframes into one
data_frames = [confirmed_tidy, deaths_tidy, recovered_tidy]          
df_corona = reduce(lambda left, right: pd.merge(
    left, right, on = id_list+['Date'], how='outer'), data_frames)


# In[6]:


# 1.4 Each row should only represent one observation
id_vars = df_corona.columns[:5]
data_type = ['Confirmed', 'Deaths', 'Recovered']
df_corona = pd.melt(df_corona, id_vars=id_vars, value_vars=data_type, var_name='Situation', value_name='Count')
df_corona['Date'] = pd.to_datetime(df_corona['Date'],format='%m/%d/%y', errors='raise').dt.date

df_corona=df_corona.sort_values(by=['Country/Region', 'Date'], ascending=True)


# In[7]:


# Automatically setting dates from loaded data
start_date = df_corona.iloc[0:]['Date'].values[0]
end_date = df_corona.iloc[-1:]['Date'].values[0]

#this is not really correct, because provinces can be separated. Will add that later
df_corona= df_corona.groupby(['Country/Region','Situation', 'Date'],as_index=False).agg({'Count':'sum'})


# In[8]:

#### Sum of global cases
sum_situations= df_corona.groupby(['Situation', 'Date'],as_index=False).agg({'Count':'sum'})

#calculate change 
sum_situations['changeRatio'] = sum_situations['Count'].pct_change(fill_method = 'pad') 

#clipping values below zero
sum_situations.changeRatio=sum_situations.changeRatio.mask(sum_situations.changeRatio.lt(0),'NaN')
#sum_situations = sum_situations.replace(np.nan, '', regex=True)

# In[10]:

#Lets average across 5 days
window_size= 5 #moving mean using 5 days data to remove jumps
sum_situations['movingMean_of_Change'] = (sum_situations['changeRatio'].rolling(min_periods=4, center=True, window=window_size)).mean()


# In[11]:
#Moving average date for last 15 days
goBackby = 15 #days
start_date2 = sum_situations.iloc[-goBackby:]['Date'].values[0]
mask = (sum_situations['Date'] >= start_date2) & (sum_situations['Date']  <= end_date)
sum_situations_cropped = sum_situations.loc[mask]

#Parameters settings
global rateICU  # Appx. percentage of patience that may need ICU
#rateICU=float(0.02) 
#prob_of_transmission=float(st.sidebar.number_input('Transmission', min_value=0.00, max_value=1.000, value=0.02, step=0.01,key=None))


#Select country and crop data to confirmed cases
@st.cache
def selectCountry(Country, df_corona):
    mask = (df_corona['Country/Region'] == Country) & (df_corona['Count'] >= 1)#first case
    df_corona_country = df_corona.loc[mask]
    return df_corona_country


def retrieveParameters(df_corona_country,start_date3):
    mask = df_corona_country['Situation']=='Recovered'
    Total_recovered = df_corona_country.loc[mask]['Count'].max()

    #infected & recovered at time when lockdown starts
    rowForDispay=df_corona_country[(df_corona_country['Date']==start_date3) & (df_corona_country['Situation']=='Confirmed')]
    Total_confirmed_Lockdownt0 = rowForDispay.Count.max()
    if np.isnan(Total_confirmed_Lockdownt0):
        Total_confirmed_Lockdownt0 =1
    
    rowForDispay=df_corona_country[(df_corona_country['Date']==start_date3) & (df_corona_country['Situation']=='Recovered')]
    Total_recovered_Lockdownt0 = rowForDispay.Count.max()

    if np.isnan(Total_recovered_Lockdownt0):
        Total_recovered_Lockdownt0 =1

    mask=(df_corona_country['Situation'] == 'Confirmed')
    df_corona_countryConf = df_corona_country.loc[mask]

    return df_corona_countryConf, Total_recovered, Total_recovered_Lockdownt0, Total_confirmed_Lockdownt0

@st.cache
def SetParameters(Days, Confirmed_in_country, Critically_ill, I0, Rec, start_date3, population):
    d2r=14                  #amount of days to recover & build immunity
    sigma = 1.0 / 2         #How quickly someone becomes infective(basically within incubation period, say day 2)   
    gamma = 1./d2r                 #mean recovery rate, gamma, (in 1/days)
    S0 = population - I0 - Rec     # Everyone else, S0, is susceptible to infection initially.
    y0=S0, I0, Rec

    t = np.linspace(0,Days , Days+1)     # A grid of time points (in days)
    
    #pad confirmed data to match lengths for plotting
    Confirmed_in_country = list(pad(Confirmed_in_country,len(t),''))
    Critically_ill = list(pad(Critically_ill,len(t),'')) #Probability from confirmed cases
        
    SimulationDatesRange=pd.date_range(start_date3, periods=len(t), freq='D')
    end_date2=SimulationDatesRange.max() # new date axis for projected simulation
    return y0, t, sigma, gamma, population, Confirmed_in_country, Critically_ill, SimulationDatesRange, end_date2


from SIReqns import SIReqns    
# Integrate the SIR equations over the time grid, t.
def calcdiffeqn(y0,t, population, beta, gamma):
    ret = odeint(SIReqns, y0, t, args=(population, beta, gamma))
    S, I, R = ret.T
    return S, I, R


def tooltips():
  return {'config': {'mark': {'tooltip': {'content': 'encoding'}}}}

alt.themes.register('tooltips', tooltips)
alt.themes.enable('tooltips')

expectedChartTitle_placeholder = st.empty()
expectedChart_placeholder = st.empty()
LockDate_placeholder = st.sidebar.empty()
fitChart_placeholder = st.sidebar.empty()

S=st.sidebar.slider('Bring Confirmed and Expected closer ↑↑', min_value=4.5, max_value=40.0, value=19.0, step=0.5, format=None)

#st.write('Our task: take the orange hill away from blue snake')

criticalOnly=st.checkbox('Critical only')

if st.checkbox('Linear scale ("curve flattening")'):
    plotScale = 'linear'
else:
    plotScale = 'log'
 
selectedCountry= st.selectbox("Country: ",df_corona['Country/Region'].unique().tolist(),index=54)
Contacts=st.slider('Contacts (take the hills away from green serpent) ', min_value=4.5, max_value=40.0, value=S, step=0.5, format=None)
#mk0=('<span style="color:#E24668;font-weight: bold; font-size: 100%">Slide to take the hills away from blue snake</span>')
#st.markdown(mk0,unsafe_allow_html=True)

expectedChartTitle_placeholder1 = st.empty()
expectedChartTitle_placeholder2 = st.empty()

if Contacts <=6:
    projVal =1600
else:
    projVal =600
    
projectionDays=st.sidebar.number_input('Days to project in future', min_value=5, max_value=2000, value=projVal, step=10,key=None)
ICUbeds=st.sidebar.number_input('Acute care units(ICUs) per 100k', min_value=0.0, max_value=10000.0, value=14.6,key=None)
rateICU=float(st.sidebar.number_input('Critical prob.', min_value=0.00, max_value=1.000, value=0.02, step=0.01,key=None))

#st.sidebar.markdown("Select log to show critical cases")        
      
mk3=(<a href="https://en.wikipedia.org/wiki/List_of_countries_by_hospital_beds" target="_blank">List of ICUs</a>)
st.sidebar.markdown(mk3,unsafe_allow_html=True)


def getContactFunc(Contacts, selectedCountry, ICUbeds):
    
    population = getPopulation.get_population(selectedCountry)
    
    #Specific country data
    df_corona_country=selectCountry(selectedCountry, df_corona)
    
    #assign new date to see the effect of lockdown
    start_date3= LockDate_placeholder.selectbox("Choose the lockdown date",df_corona_country['Date'].unique().tolist(),index=0)   
    
    df_corona_countryConf, Total_recovered, Total_recovered_Lockdownt0, Total_confirmed_Lockdownt0=\
        retrieveParameters(df_corona_country,start_date3)

    mask = (df_corona_countryConf['Date']>= start_date3)
    df_corona_countryConf = df_corona_countryConf.loc[mask]

    Confirmed_in_country = df_corona_countryConf['Count'].values.tolist()
    Critically_ill = [i * rateICU for i in Confirmed_in_country]
    
    y0, t, sigma, gamma, population, Confirmed_in_country, Critically_ill, SimulationDatesRange, end_date2 = \
        SetParameters(projectionDays,Confirmed_in_country, Critically_ill, Total_confirmed_Lockdownt0, Total_recovered_Lockdownt0, start_date3, population)    

    prob_of_transmission = 0.018 # from literature, leave it as it is.
    
    # Contact rate beta = Probability of transmission x Number of contacts
    beta = np.around((prob_of_transmission*Contacts), decimals=4) #beta = 0.36 # early italian estimate
    
    S, I, R = np.around((calcdiffeqn(y0, t, population, beta, gamma)),decimals=2)
    R0 = beta / gamma     #R0 Value
    
    totalICUbeds=((ICUbeds/100000)*population)/2    # 50% occupancy, change this if you have total nrs.
    
    
    #From Paul E. Parham, Edwin Michael. Outbreak properties of epidemic models: 
    #The roles of temporal forcing and stochasticity on pathogen invasion dynamics. 
    #Journal of Theoretical Biology, Elsevier, 2011, 271 (1), pp.1. 10.1016/j.jtbi.2010.11.015 . hal-00657584
    s1 = 0.5 * (-(sigma + gamma) + math.sqrt((sigma + gamma) ** 2 + 4 * sigma * gamma * (R0 -1)))
    
    print("Doubling every ~%.1f" % (math.log(2.0, math.e) / s1), "days with R0", np.around(R0, decimals=2))
    print('beta(contact rate): ', beta)
    print('Population of ',selectedCountry,': ',population)
    
    #rounding 
    Critically_ill_exp=list(map(round, (I*rateICU)))
    I=list(map(round, (I)))

    #no padding needed
    Critically_ill_exp = list(pad(Critically_ill_exp,len(t),'')) #Probability from expected cases
    
    #putting everything to a common date
    #df_model= pd.DataFrame({'Date':SimulationDatesRange, 'Susceptible':S, 'Expected':I, 'RecoveredWithImmunity':R, 'Confirmed':Confirmed_in_country, 'Critical':Critically_ill_exp})
    df_model= pd.DataFrame({'Date':SimulationDatesRange, 'Expected':I,'Confirmed':Confirmed_in_country, 'Critical':Critically_ill_exp})
    tidy_df_model = df_model.melt(id_vars=["Date"])
    
    ##Model vs confirmed cases entire life cycle
    #renamed dataframe for consistency
    tidy_df_model = tidy_df_model.rename({'value': 'Count', 'variable': 'Situation'}, axis=1)
    
    tidy_df_model['ICUs'] = np.int(totalICUbeds)
    tidy_df_model['ICUsText'] = ('ICUs @50% Occup.: '+np.str(np.int(totalICUbeds)))

    
    #Droppping unnecessary rows for plot
    mask = (tidy_df_model['Situation']!='Confirmed') & (tidy_df_model['Situation']!='Critical')
    OnlyExpected = tidy_df_model.loc[mask]
    
    #Droppping unnecessary rows for plot
    mask = (tidy_df_model['Situation']!='Expected') & (tidy_df_model['Situation']!='Confirmed')
    OnlyCritical = tidy_df_model.loc[mask]
    
    #Droppping unnecessary rows for plot
    mask = (tidy_df_model['Situation']!='Expected') & (tidy_df_model['Situation']!='Critical')
    OnlyConfirmed= tidy_df_model.loc[mask]
    
    ##Model vs confirmed cases until now
    mask = (tidy_df_model['Date'].dt.date>= start_date3) & (tidy_df_model['Date'].dt.date<= end_date) 
    df_corona_country_cropped = tidy_df_model.loc[mask]
    
    return OnlyConfirmed, OnlyExpected, OnlyCritical,df_corona_country_cropped, tidy_df_model, totalICUbeds, population, start_date3


#call contactFunction
OnlyConfirmed, OnlyExpected, OnlyCritical,df_corona_country_cropped, tidy_df_model, totalICUbeds, population, start_date3=getContactFunc(Contacts,selectedCountry, ICUbeds)
#insert this for a horizontal line indicating ICUs in the country
scaleFactorC = 0.013
scaleFactor = 0.7

rowForDispay=OnlyExpected[OnlyExpected['Count']==OnlyExpected['Count'].max()]
infectedPeakCount=np.str(np.int(rowForDispay.Count.max()))
infectedPeakDate = np.str(rowForDispay.Date.dt.date.max())


rowForDispay2=OnlyCritical[OnlyCritical['Count']==OnlyCritical['Count'].max()]
criticalPeakCount=np.str(np.int(rowForDispay2.Count.max()))
criticalPeakDate = np.str(rowForDispay2.Date.dt.date.max())


OnlyConfirmed=OnlyConfirmed.replace(r'^\s*$', np.nan, regex=True)
rowForDispay3=OnlyConfirmed[OnlyConfirmed['Count']==OnlyConfirmed['Count'].max()]
confirmedPeakCount=np.str(np.int(rowForDispay3.Count.max()))
confirmedPeakDate = np.str(rowForDispay3.Date.dt.date.max())

mk1=(f'<div><span style="color:#26272F;font-weight: bold; font-size: 100%">Peaks on: {infectedPeakDate}</span></div> ' +f'<span style="color:#F99E4C;font-weight: bold; font-size: 100%">Infected: ~{infectedPeakCount}</span>'+f'<div><span style="color:#EF4648;font-weight: bold; font-size: 100%">Critical: ~{criticalPeakCount}</span></div>'+f'<span style="color:#3F6A8A;font-weight: bold; font-size: 100%">Confirmed: ~{confirmedPeakCount}</span>')

expectedChartTitle_placeholder.markdown(mk1,unsafe_allow_html=True)


ExpCountryPlot= alt.Chart(OnlyExpected).transform_filter(alt.datum.Count>0.01).mark_area(clip=True,opacity=0.7).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactor*population)), axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color =alt.value("#F99E4C"),
                                                                  tooltip=(['Date', 'Count'])
                                                                ).interactive()

CriticalCountryPlot0= alt.Chart(OnlyCritical).transform_filter(alt.datum.Count>0.01).mark_area(clip=True,opacity=0.7).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactorC*population)), axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color =alt.value("#EF4648"),
                                                                  tooltip=['Date', 'Count']
                                                                ).interactive()

ConfirmedCountryPlot= alt.Chart(OnlyConfirmed).transform_filter(alt.datum.Count>0.01).mark_line(size=6,clip=True, opacity=0.9).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactorC*population)), axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color =alt.value("#06A108"),
                                                                  tooltip=['Date', 'Count']
                                                                ).interactive()


ICULine = alt.Chart(OnlyCritical).mark_rule(color="#26272F", opacity=0.3, size=5,strokeDash=[1, 1]).encode(
    #x='Date:T',
    y="ICUs:Q",
    size=alt.value(3)
)

annotationICU = alt.Chart(OnlyCritical).mark_text(
    align='left',
    baseline='bottom',
    fontSize = 14,
    dx = 0,
    color='#26272F'
).encode(
    y='ICUs',
    text='ICUsText')  
    
if criticalOnly:
    expectedChart_placeholder.altair_chart((CriticalCountryPlot0+ICULine+annotationICU),use_container_width=True)
else:
    if plotScale =='log':
        expectedChart_placeholder.altair_chart((ExpCountryPlot+CriticalCountryPlot0+ICULine+annotationICU+ConfirmedCountryPlot),use_container_width=True)
    else:
        expectedChart_placeholder.altair_chart((ExpCountryPlot+CriticalCountryPlot0+ICULine+annotationICU),use_container_width=True)



expectedChartTitle_placeholder1.text('⇣')
expectedChartTitle_placeholder2.text('⇣')

countryPlot= alt.Chart(df_corona_country_cropped).mark_point(clip=True, filled=True,size=100, opacity=0.9).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', legend = alt.Legend(title = '', orient ='top',labelFontSize=12)),
                                                                  tooltip=['Date', 'Count']
                                                                 ).interactive().configure(background='transparent')


#st.sidebar.altair_chart((countryPlot), use_container_width=True)

fitChart_placeholder.altair_chart(countryPlot,use_container_width=True)


# ICULine = alt.Chart(df_corona_country_cropped).mark_rule(color="red", strokeDash=[2, 1]).encode(
#     y="ICUs:Q",
#     size=alt.value(3)
# )

#st.altair_chart(points, use_container_width=True)
st.markdown('Global Growth:')
globalPlot= alt.Chart(sum_situations).mark_area().encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', legend = alt.Legend(title = '', orient ='top',labelFontSize=12)),
                                                                ).properties(width=270,height=300).interactive().configure(background='transparent')

st.altair_chart(globalPlot,use_container_width=True)


st.markdown('Change:')
d= alt.Chart(sum_situations).mark_area().encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Change')),
                                                                  color = alt.Color('Situation:N', legend = None),
                                                                ).properties(width=270,height=300).interactive().configure(background='transparent')
st.altair_chart(d,use_container_width=True)

st.markdown('Change in last 15 days')
d= alt.Chart(sum_situations_cropped).mark_area().encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Change')),
                                                                  color = alt.Color('Situation:N', legend = None),
                                                                ).properties(width=270,height=300).interactive().configure(background='transparent')
st.altair_chart(d,use_container_width=True)



mk4=(<a href="https://www.linkedin.com/in/diwakerzha/" target="_blank">List of ICUs</a>)
st.markdown(mk4,unsafe_allow_html=True)


#TotalConfStart_text = st.sidebar.empty()
#TotalConfEnd_text = st.sidebar.empty()
TotalRate_text = st.sidebar.empty()

#Wordlwide Rough approximation of growth rate (this excludes all variables such as excluding immunization, and suppressions tactics)
Situation = 'Confirmed' # change to the topic you are interested to calculate rate.
N_t0 = sum_situations[(sum_situations.Date==start_date)&(sum_situations.Situation == Situation)]['Count'].values[0]
#print('Total confirmed cases on', start_date, '=',N_t0)

N_tD = sum_situations[(sum_situations.Date==end_date)&(sum_situations.Situation == Situation)]['Count'].values[0]
#print('Total confirmed cases on',end_date, '=', N_tD)

#Number of days since t=0 (this is considering cropped time)
D = (end_date-start_date).days
#print('Days since t0 =', D)

#IR = infection rate
IR = np.exp(np.log(N_tD/N_t0)/D)
#print('Infection rate =', IR)


