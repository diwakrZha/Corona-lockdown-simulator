# Corona case predictor
 Streamlit app which helps you calculate the effect of lockdown in the wake of CoViD-19 (Corona) virus pandemic
 ![](usage.gif)
 
The webapp was created with a single SEIR function, which is useful for early days but does not represent well when a dynamic and varying degree of social distancing and vaccinations are in place. A sum of multiple SEIR functions with a range of parameters, either learned or obtained from a fit to observed data may give a better estimate. Here is the source code, use it the way you like to improve it :)

 1. You can move the contacts slider to see the affect of reduced interaction on total infection and critical case numbers.
 2. You can see how under resourced the country is in terms of ICUs.
 3. In the sidebar choose the lockdown date (or any date if you want a short term windowed analysis).
 4. Chart shows the "Expected" infections from the SIR model and real world "Confirmed" infections.
 5. Move the slider so that the circles for Expected and Confirmed come as close as possible to eachother (I will later replace this with curve_fit).
 6. The value will automatically be transferred to the main UI window "Contacts" slider.
 7. Italy's original (before lockdown),the number for contacts was ~19 (compare how this compares with the number of "Contacts" after lockdown)
 8. Now move the "Contacts" slider on the main UI page to estimate how long this may last and how many can get critically ill.
 9. Be safe everyone.
 
The app is using data from John Hopkins University's public repository.

Here is my article on medium that uses this repository:
https://diwaker-phd.medium.com/how-long-will-the-corona-lockdown-last-8f23ef1730aa



I was inspired from these repositories:

 https://github.com/YiranJing/Coronavirus-Epidemic-COVID-19
 
 https://github.com/pdtyreus/coronavirus-ds
 
 https://github.com/CSSEGISandData/COVID-19
 
 https://github.com/amtam0/coronavirus-world-map
 
 and more snippets from stackoverflow and streamlit docs.
