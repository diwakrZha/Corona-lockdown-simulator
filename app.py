#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:10:09 2020

@author: diwaker
"""

import pandas as pd
import numpy as np
import altair as alt
from scipy.integrate import odeint
from itertools import chain, repeat, islice
import math
import getPopulation
from datetime import timedelta

import streamlit as st

#Some book keeping, will be needed later
def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

st.title('Corona Lockdown')



#Downloading data from JHU repository
@st.cache
def downloadData():
    df_corona = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')
    return df_corona

df_corona=downloadData()

df_corona= df_corona.groupby(['Country/Region', 'Date'],as_index=False).sum()
df_corona=df_corona.drop(['Lat', 'Long'], axis=1)

df_corona = pd.melt(df_corona, id_vars=['Country/Region', 'Date'], value_vars=['Confirmed', 'Deaths'])

# Automatically setting dates from loaded data
start_date = df_corona.iloc[0:]['Date'].values[0]
end_date = df_corona.iloc[-1:]['Date'].values[0]

#this is not really correct, because provinces can be separated. Will add that later
df_corona= df_corona.groupby(['Country/Region','variable', 'Date'],as_index=False).agg({'value':'sum'})
df_corona = df_corona.rename({'value': 'Count', 'variable': 'Situation'}, axis=1)

# In[8]:

#### Sum of global cases
sum_situations= df_corona.groupby(['Situation', 'Date'],as_index=False).agg({'Count':'sum'})

#calculate change 
sum_situations['changeRatio'] = sum_situations['Count'].pct_change(periods=2, fill_method='pad', limit=None, freq=None) 

#clipping values below zero
sum_situations.changeRatio=sum_situations.changeRatio.mask(sum_situations.changeRatio.lt(0),'NaN')
#sum_situations = sum_situations.replace(np.nan, '', regex=True)

#Lets average across 5 days
window_size= 5 #moving mean using 5 days data to remove jumps
sum_situations['movingMean_of_Change'] = (sum_situations['changeRatio'].rolling(min_periods=1, center=True, window=window_size)).mean()


# In[11]:
#Moving average date for last 15 days
goBackby = 20 #days
start_date2 = sum_situations.iloc[-goBackby:]['Date'].values[0]
mask = (sum_situations['Date'] >= start_date2) & (sum_situations['Date']  <= end_date)
sum_situations_cropped = sum_situations.loc[mask]

#Parameters settings
global rateICU  # Appx. percentage of patience that may need ICU

#Select country and crop data to confirmed cases
@st.cache
def selectCountry(Country, df_corona):
    mask = (df_corona['Country/Region'] == Country) & (df_corona['Count'] >= 1)#first case
    df_corona_country = df_corona.loc[mask]
    return df_corona_country


def retrieveParameters(df_corona_country,start_date3):
    #mask = (df_corona_country['Situation']=='Recovered') & (df_corona_country['Date']==start_date3)
    Total_recovered = 1#df_corona_country.loc[mask]['Count'].max()
    
    
    mask = (df_corona_country['Situation']=='Deaths') & (df_corona_country['Date']==start_date3)
    D0 = df_corona_country.loc[mask]['Count'].max()
    if np.isnan(D0):
        D0 =0
    
    mask = (df_corona_country['Situation']=='Deaths')
    OnlyDeaths = df_corona_country.loc[mask]['Count']
        
    #infected & recovered at time when lockdown starts
    rowForDispay=df_corona_country[(df_corona_country['Date']==start_date3) & (df_corona_country['Situation']=='Confirmed')]
    Total_confirmed_Lockdownt0 = rowForDispay.Count.max()
    if np.isnan(Total_confirmed_Lockdownt0):
        Total_confirmed_Lockdownt0 =1
    
    #rowForDispay=df_corona_country[(df_corona_country['Date']==start_date3) & (df_corona_country['Situation']=='Recovered')]
    Total_recovered_Lockdownt0 = 1#rowForDispay.Count.max()

    if np.isnan(Total_recovered_Lockdownt0):
        Total_recovered_Lockdownt0 =1

    mask=(df_corona_country['Situation'] == 'Confirmed')
    df_corona_countryConf = df_corona_country.loc[mask]

    return df_corona_countryConf, Total_recovered, Total_recovered_Lockdownt0, D0, Total_confirmed_Lockdownt0, OnlyDeaths

#@st.cache
def SetParameters(Days, Confirmed_in_country, Critically_ill, E0, D0, Rec, start_date3, population):
    d2r=14              #amount of days to recover & build immunity
    sigma = 1.0 / 2         #How quickly someone becomes infective(basically within incubation period, say day 2)   
    gamma = 1./d2r                 #mean recovery rate, gamma, (in 1/days)
    S0 = population - E0 - Rec-D0     # Everyone else, S0, is susceptible to infection initially.
    I0=E0-Rec-D0

    y0=S0, E0, I0, Rec
    #st.write(E0,I0, D0, Rec)
    t = np.linspace(0,Days , Days+1)     # A grid of time points (in days)
    
    #pad confirmed data to match lengths for plotting
    Confirmed_in_country = list(pad(Confirmed_in_country,len(t),''))
    Critically_ill = list(pad(Critically_ill,len(t),'')) #Probability from confirmed cases
        
    SimulationDatesRange=pd.date_range(start_date3, periods=len(t), freq='D')
    end_date2=SimulationDatesRange.max() # new date axis for projected simulation
    return y0, t, sigma, gamma, population, Confirmed_in_country, D0, Critically_ill, SimulationDatesRange, end_date2


from SEIReqns import SEIReqns
# Integrate the SIR equations over the time grid, t.
def calcdiffeqn(y0,t, population, SDMod, beta, gamma, sigma):
    ret = odeint(SEIReqns, y0, t, args=(population, SDMod, beta, gamma, sigma))
    S, E, I, R = ret.T
    return S, E, I, R


def tooltips():
  return {'config': {'mark': {'tooltip': {'content': 'encoding'}}}}

alt.themes.register('tooltips', tooltips)
alt.themes.enable('tooltips')

expectedChartTitle_placeholder = st.empty()
expectedChart_placeholder = st.empty()
LockDate_placeholder = st.sidebar.empty()

st.sidebar.markdown('**Calibrate model to confirmed data**')
fitChart_placeholder = st.sidebar.empty()

ListCountries = df_corona['Country/Region'].unique().tolist()
defCountry= ListCountries.index('Germany')
selectedCountry= st.selectbox("Country: ",ListCountries,index=defCountry)

S=st.sidebar.slider('Calibration: bring \"Confirmed\" & \"Expected\" closer', min_value=4.5, max_value=40.0, value=20.50, step=0.5, format=None)
confirmedCase_placeholder = st.empty()

Contacts=st.slider('< Less - | Social interactions [a.u.] | - More >', min_value=4.5, max_value=40.0, value=S, step=0.5, format=None)

criticalOnly=st.checkbox('Only Show estimate for critically ill')

if st.checkbox('Linear scale ("flattening the curve")'):
    plotScale = 'linear'
    yAxisName= ('COUNT (linear scale)')
else:
    plotScale = 'log'
    yAxisName= ('COUNT (log. scale)')


st.write('To reduce deaths, hills should spread out & the blue line should bend down')
#mk0=('<span style="color:#E24668;font-weight: bold; font-size: 100%">Slide to take the hills away from blue snake</span>')
#st.markdown(mk0,unsafe_allow_html=True)

expectedChartTitle_placeholder1 = st.empty()
expectedChartTitle_placeholder2 = st.empty()

if Contacts <=6:
    projVal =1800
else:
    projVal =600
    
projectionDays=projVal#st.sidebar.number_input('Days to project in future', min_value=5, max_value=2000, value=projVal, step=10,key=None)
ICUbeds=st.sidebar.number_input('Acute care units(ICUs) per 100k', min_value=0.0, max_value=10000.0, value=29.2,key=None)
rateICU=float(st.sidebar.number_input('Critical illness rate [~ 0.7 to 6 %]', min_value=0.05, max_value=100.0, value=1.0, step=0.05,key=None))
rateICU=rateICU/100
#st.sidebar.markdown("Select log to show critical cases")        
      
mk3=('<a href="https://en.wikipedia.org/wiki/List_of_countries_by_hospital_beds" target="_blank">List of ICUs</a>')
st.sidebar.markdown(mk3,unsafe_allow_html=True)

description=('<a href="https://medium.com/@diwaker.jha/how-long-will-the-corona-lockdown-last-8f23ef1730aa" target="_blank">Usage tutorial</a>')
st.sidebar.markdown(description,unsafe_allow_html=True)


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def getContactFunc(Contacts, selectedCountry, ICUbeds):
    
    population = getPopulation.get_population(selectedCountry)
    
    #Specific country data
    df_corona_country=selectCountry(selectedCountry, df_corona)
    #st.write(df_corona_country)
    #assign new date to see the effect of lockdown
    start_date3= LockDate_placeholder.selectbox("Choose the lockdown date",df_corona_country['Date'].unique().tolist(),index=0)       
    
    df_corona_countryConf, Total_recovered, Total_recovered_Lockdownt0, D0, Total_confirmed_Lockdownt0, OnlyDeaths=\
        retrieveParameters(df_corona_country,start_date3)

    mask = (df_corona_countryConf['Date']>= start_date3)
    
    df_corona_countryConf = df_corona_countryConf.loc[mask]

    Confirmed_in_country = df_corona_countryConf['Count'].values.tolist()
    Critically_ill = [i * rateICU for i in Confirmed_in_country]
    
    y0, t, sigma, gamma, population, Confirmed_in_country, D0, Critically_ill, SimulationDatesRange, end_date2 = \
        SetParameters(projectionDays,Confirmed_in_country, Critically_ill, Total_confirmed_Lockdownt0, D0,Total_recovered_Lockdownt0, start_date3, population)    
        
    prob_of_transmission = 0.018 # from literature, leave it as it is.
    #Say R0= 2.5 # as seen
    #beta = 0.4 #Italy (0.37, more here 24 January–8 February [Median (95% CIs)], https://science.sciencemag.org/content/early/2020/03/24/science.abb3221/tab-figures-data)
    SDMod = 1
    #beta=Contacts*beta #Social Distancing modulator (normalizing to 100% being normal) 
    
    # Contact rate beta = Probability of transmission x Number of contacts
    beta = np.around((prob_of_transmission*Contacts), decimals=4) #beta = 0.36 # early italian estimate
    
    
    # Contact rate beta = Probability of transmission x Number of contacts
    #beta = np.around((beta*SDMod), decimals=4) #beta = 0.36 # early italian estimate
    #beta = 0.36*Contacts
    S, E, I, R = calcdiffeqn(y0,t, population, SDMod, beta, gamma, sigma)
    #st.write(S,E, I, R)
    R0 = (beta*SDMod) / gamma     #R0 Value
    #st.write(beta*SDMod)
    #st.write(R0)
    
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
    mask = (tidy_df_model['Date'] >= start_date3) & (tidy_df_model['Date'] <= end_date) 
    df_corona_country_cropped = tidy_df_model.loc[mask]
#    
#    OnlyExpectedSeries = OnlyExpected['Count'] # convert to pd.Series
#    OnlyConfirmedSeries = OnlyConfirmed['Count'] # convert to pd.Series
#    # start with first infections
#
#    OnlyExpectedSeries = OnlyExpectedSeries[OnlyExpectedSeries.values != 0]
#    OnlyConfirmedSeries = OnlyConfirmedSeries[OnlyConfirmedSeries.values != 0]
#    OnlyConfirmedSeries = OnlyConfirmedSeries[OnlyConfirmedSeries.values != '']
#    
#    
#    poptimal_exponentialE, pcovariance_exponentialE = curve_fit(exponential, np.arange(len(OnlyExpectedSeries.values)), OnlyExpectedSeries, p0=[0.3, 0.205, 0])
#    poptimal_exponentialC, pcovariance_exponentialC = curve_fit(exponential, np.arange(len(OnlyConfirmedSeries.values)), OnlyConfirmedSeries, p0=[0.3, 0.205, 0])
#    
#    st.write(poptimal_exponentialE)
#    st.write(poptimal_exponentialC)


    return selectedCountry, OnlyConfirmed, OnlyExpected, OnlyCritical,df_corona_country_cropped, tidy_df_model, totalICUbeds, population, start_date3, OnlyDeaths, df_corona_country

#call contactFunction
selectedCountry, OnlyConfirmed, OnlyExpected, OnlyCritical,df_corona_country_cropped, tidy_df_model, totalICUbeds, population, start_date3, OnlyDeaths, df_corona_country=getContactFunc(Contacts,selectedCountry, ICUbeds)
#insert this for a horizontal line indicating ICUs in the country
scaleFactorC = 0.006
scaleFactor = 0.555

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

mk1=(f'<span style="color:#F99E4C;font-weight: bold; font-size: 100%">Infections: ~{infectedPeakCount}</span>'+f'<div><span style="color:#EF4648;font-weight: bold; font-size: 100%">Critical: ~{criticalPeakCount}</span></div>'+f'<div><span style="color:#26272F;font-weight: bold; font-size: 100%">Peaks on: {infectedPeakDate}</span></div> ')

expectedChartTitle_placeholder.markdown(mk1,unsafe_allow_html=True)


Total_dead = OnlyDeaths.max()
if np.isnan(Total_dead):
    Total_dead =0 # until now

OnlyConfirmed ['newCases'] = OnlyConfirmed.Count.diff()
try:
    two_weeks_ago = pd.to_datetime(end_date) - timedelta(weeks = 2) #correct formula for CFR
    CFR=(np.around(Total_dead/(OnlyConfirmed.loc[OnlyConfirmed["Date"] == two_weeks_ago, "Count"]).iloc[0],decimals=2))*100
except:
    CFR ='NA'


mk5=(f'<span style="color:#5677A4;font-weight: bold; font-size: 100%">Confirmed: {confirmedPeakCount}</span>'+' | 'f'<span style="color:red;font-weight: bold; font-size: 100%">Deaths: {Total_dead}</span>')
confirmedCase_placeholder.markdown(mk5,unsafe_allow_html=True)



ExpCountryPlot= alt.Chart(OnlyExpected).transform_filter(alt.datum.Count>0.01).mark_area(clip=True,opacity=0.7).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'DATE',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactor*population)), axis = alt.Axis(title = yAxisName,format = ("~s"))),
                                                                  color =alt.value("#F99E4C"),
                                                                  tooltip=(['Date', 'Count'])
                                                                ).interactive()

CriticalCountryPlot0= alt.Chart(OnlyCritical).transform_filter(alt.datum.Count>0.01).mark_area(clip=True,opacity=0.7).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'DATE',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactorC*population)), axis = alt.Axis(title = yAxisName,format = ("~s"))),
                                                                  color =alt.value("#EF4648"),
                                                                  tooltip=['Date', 'Count']
                                                                ).interactive()

ConfirmedCountryPlot= alt.Chart(OnlyConfirmed).transform_filter(alt.datum.Count>0.01).mark_line(size=6,clip=True, opacity=0.9).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'DATE',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q',scale=alt.Scale(type=plotScale,domain=(1,scaleFactorC*population)), axis = alt.Axis(title = yAxisName,format = ("~s"))),
                                                                  color =alt.value("#5677A4"),
                                                                  tooltip=['Date', 'Count']
                                                                ).interactive()


ICULine = alt.Chart(OnlyCritical).mark_rule(color="green", opacity=0.3, size=5,strokeDash=[1, 1]).encode(
    #x='Date:T',
    y="ICUs:Q",
    size=alt.value(3)
)

annotationICU = alt.Chart(OnlyCritical).mark_text(
    align='left',
    baseline='bottom',
    fontSize = 14,
    dx = 0,
    color='green'
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


mk6=(f'<span style="color:red;font-weight: bold; font-size: 100%">Current crude fatality rate (T-14): {CFR}</span>')

#expectedChartTitle_placeholder1.markdown(mk6,unsafe_allow_html=True)
expectedChartTitle_placeholder2.text('⇣') #F99E4C

countryPlot= alt.Chart(df_corona_country_cropped).mark_line(clip=True, point=True, size =5, opacity=0.6).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date',format = ("%b %Y"))),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', 
                                                                                    scale=alt.Scale(domain=['Confirmed', 'Expected','Critical'],
                                                                                                    range=['#5677A4', '#F99E4C','#EF4648']),
                                                                                    legend = alt.Legend(title = '',orient ='bottom',labelFontSize=12)),
                                                                  tooltip=['Date', 'Count']
                                                                 ).interactive().configure(background='transparent')


fitChart_placeholder.altair_chart(countryPlot,use_container_width=True)

df_corona_countryChange = df_corona_country.copy()
#calculate change 
df_corona_countryChange['changeRatio'] = df_corona_countryChange['Count'].pct_change(periods=2, fill_method='pad', limit=None, freq=None) 

#clipping values below zero
df_corona_countryChange.changeRatio=df_corona_countryChange.changeRatio.mask(df_corona_countryChange.changeRatio.lt(0),'NaN')
#sum_situations = sum_situations.replace(np.nan, '', regex=True)

#Lets average across 5 days
window_size= 5 #moving mean using 5 days data to remove jumps
df_corona_countryChange['movingMean_of_Change'] = (df_corona_countryChange['changeRatio'].rolling(min_periods=1, center=True, window=window_size)).mean()


# In[11]:
#Moving average date for last 15 days
goBackby = 20#days
start_date2 = df_corona_countryChange.iloc[-goBackby:]['Date'].values[0]
mask = (df_corona_countryChange['Date'] >= start_date2) & (df_corona_countryChange['Date']  <= end_date)
df_corona_country_croppedChange = df_corona_countryChange.loc[mask]


#st.altair_chart(points, use_container_width=True)
st.markdown('**Country specific growth:**')
st.markdown(selectedCountry)


d0= alt.Chart(df_corona_countryChange).mark_area(opacity =0.5).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', scale=alt.Scale(domain=['Confirmed', 'Deaths'],
                                                                                                    range=['#5677A4', 'red']),
                                                                                    legend = alt.Legend(title = 'Cases', orient ='top',labelFontSize=12)),
                                                                ).interactive()

d1= alt.Chart(df_corona_countryChange[df_corona_countryChange['Situation']=='Confirmed']).mark_area(color='green', size=100, filled=True, opacity=0.1).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Change')),
                                                                  opacity=alt.Opacity('Situation', legend=alt.Legend(title="Rate", orient='right'))

                                                                  #color = alt.Color(scale=alt.Scale(range=['red']), legend = alt.Legend(title = 'Rate', orient ='right',labelFontSize=12)),
                                                                ).interactive()


d3=(d0+d1).resolve_scale(y='independent').properties(width=300,height=400).interactive()
st.altair_chart(d3,use_container_width=True)

st.markdown('**Growth rate in last 20 days:**')

d0= alt.Chart(df_corona_country_croppedChange).mark_area(opacity =0.5).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', scale=alt.Scale(domain=['Confirmed', 'Deaths'],
                                                                                                    range=['#5677A4', 'red']),
                                                                                    legend = alt.Legend(title = 'Cases', orient ='top',labelFontSize=12)),
                                                                ).interactive()

d1= alt.Chart(df_corona_country_croppedChange[df_corona_country_croppedChange['Situation']=='Confirmed']).mark_area(color='green', size=100, filled=True, opacity=0.1).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Change')),
                                                                  opacity=alt.Opacity('Situation', legend=alt.Legend(title="Rate", orient='right'))

                                                                  #color = alt.Color(scale=alt.Scale(range=['red']), legend = alt.Legend(title = 'Rate', orient ='right',labelFontSize=12)),
                                                                ).interactive()


d3=(d0+d1).resolve_scale(y='independent').properties(width=300,height=400).interactive()
st.altair_chart(d3,use_container_width=True)

#st.markdown('**Growth rate in last 20 days**')
#d2= alt.Chart(df_corona_country_croppedChange).mark_area().encode(
#                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
#                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Change')),
#                                                                  color = alt.Color('Situation:N', legend = None),
#                                                                ).properties(width=300,height=400).interactive()
#st.altair_chart(d2,use_container_width=True)

st.markdown('______________________________')

#st.altair_chart(points, use_container_width=True)
st.markdown('**Global growth:**')

globalPlot= alt.Chart(sum_situations).mark_area(opacity =0.5).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('Count:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', scale=alt.Scale(domain=['Confirmed', 'Deaths'],
                                                                                                    range=['#5677A4', 'red']),
                                                                                    legend = alt.Legend(title = '', orient ='top',labelFontSize=12)),
                                                                ).properties(width=300,height=400).interactive()
    
st.altair_chart(globalPlot,use_container_width=True)




st.markdown('**Growth rate:**')
st.write('Wuhan, China went into a complete lockdown on 23rd of january, the infection growth rate fell to minimum on ~26th February. It increased again because other parts of the world were still open. In mid of March, most countries outside of China implemented a lockdown as well and now after two weeks we see a slow down again. The lockdown seems to be working!')

d= alt.Chart(sum_situations).mark_area(opacity =0.5).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', scale=alt.Scale(domain=['Confirmed', 'Deaths'],
                                                                                                    range=['#5677A4', 'red']),
                                                                                    legend = alt.Legend(title = '', orient ='top',labelFontSize=12)),
                                                                ).properties(width=300,height=400).interactive()
st.altair_chart(d,use_container_width=True)

    


st.markdown('Growth rate in last 20 days')

d= alt.Chart(sum_situations_cropped).mark_area(opacity =0.5).encode(
                                                                  x = alt.X('Date:T', axis = alt.Axis(title = 'Date')),
                                                                  y = alt.Y('movingMean_of_Change:Q', axis = alt.Axis(title = 'Count',format = ("~s"))),
                                                                  color = alt.Color('Situation:N', scale=alt.Scale(domain=['Confirmed', 'Deaths'],
                                                                                                    range=['#5677A4', 'red']),
                                                                                    legend = alt.Legend(title = '', orient ='top',labelFontSize=12)),
                                                                ).properties(width=300,height=400).interactive()
st.altair_chart(d,use_container_width=True)


st.markdown('______________________________')

st.markdown('I am a physicist with mathematical modelling and data expertise not an epidemiologist. This tool only visualizes projection from mathematical model using existing data. If you have corrections and/or want improvements please')
mk4=('<a href="https://www.linkedin.com/in/diwakerzha/" target="_blank">Get in touch</a>')
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

end_date=pd.to_datetime(end_date).date()
start_date=pd.to_datetime(start_date).date()

D = (end_date-start_date).days
#print('Days since t0 =', D)

#IR = infection rate
IR = np.exp(np.log(N_tD/N_t0)/D)
#print('Infection rate =', IR)
