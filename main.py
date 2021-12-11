import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##1. Real-world scenario
##The project should use a real-world dataset and include a reference of their source in the report (10)
#database source: https://www.kaggle.com/ramjasmaurya/aviation-history19702021

## 2. Importing data
## Your project should make use of one or more of the following: Relational database, API or web scraping (10)
## Import a CSV file into a Pandas DataFrame(10)
# Data used here is aviation passenger carried from 1970 to 2020 for more than 200 countries.
data1 = pd.read_csv("Passengers carried.csv")
print(data1.shape)

#3.Analysing data
print(data1.head())
# after printing 1st 5 lines of dataset, missing values are noticed
# print missing value
missing_values_count = data1.isnull().sum()
print(missing_values_count[0:53])

## Replace missing values or drop duplicates(10)
# filled missing value with 0
cleaned_data1 = data1.fillna(0)
print(cleaned_data1.shape)
print(cleaned_data1.head())

#drop the duplicated if any
drop_duplicates= cleaned_data1.drop_duplicates()
print(data1.shape,drop_duplicates.shape)
# after checking duplicated, no duplicated detected

## Slicing, loc or iloc(10)
# remove the columns where all the lines are 0
df1 = cleaned_data1.loc[:, cleaned_data1.any()]
print(data1.shape,df1.shape)
print(df1.head())
# noticing the column 2020 is removed

## sorting, indexing, and grouping(10)
# group by country to show only 1 country's passengers data
grouped_by_country = df1.groupby('Country Name')
print(grouped_by_country.get_group('China'))

# select only the row for China for the year 2015
print("2015 China's Passengers number is : ")
df1_China_2015 = df1.loc[df1["Country Name"] == "China",['Country Name', '2015']]
print(df1_China_2015)


#Import 2nd dataframe showing population by country by year
#CSV from https://www.kaggle.com/imdevskp/world-population-19602018?select=population_total_long.csv
data2 = pd.read_csv("population_total_long.csv")
print(data2.shape)
print(data2.head())

# change column name 'Count' to 'Population' for better understanding/display
data2 = data2.set_axis(["Country Name", "Year", "Population"], axis=1)
#print(data2.head())

# select for China Population in year 2015
df2_China= data2.loc[data2["Country Name"] == "China",['Country Name', 'Year', 'Population']]
df2_China_2015Population=df2_China.loc[df2_China["Year"]==2015]
print("China 2015 Population:")
print(df2_China_2015Population)

## Looping, iterrows (10)


## Merge DataFrames (10)
# Merge above 2 dataframes into 1 : mergedRes
mergedRes = pd.merge(df1_China_2015, df2_China_2015Population)
# print merged dataframe to see if displayed info is clear
print(mergedRes)

# change the order of column for better display
mergedRes=mergedRes[["Country Name", "Year", "2015","Population"]]
# change the name of column 2015 to "Passenger": here column 2015 is representing 2015 passenger number
# so this change is safe
China_2015_Passenger_Population= mergedRes.set_axis(["Country Name", "Year", "Passenger", "Population"],axis=1)
print("China 2015 Passenger and population:")
print(China_2015_Passenger_Population)

##4. Python
## Define a custom function to create reusable code (10)

## Function definition: to select a country and year for showing population info from dataframe2
def selected_country_year(country,year):
    df2_country = data2.loc[data2["Country Name"] == country, ["Country Name", "Year", "Population"]]
    df2_Country_YearPopulation = df2_country.loc[df2_country["Year"] == year]
    print(f'Population for country {country} in year {year}:')
    print(df2_Country_YearPopulation)
    return

#get user input for a country
#country_userinput = input("Please enter a country:\n")
#print(f'You entered country: {country_userinput}')
#get user input for a year
#year_userinput = input("Please enter a year:\n")
#year_userinput= int (year_userinput) # convert the str into int
#print(f'You entered year: {year_userinput}')

# call the function to show the population of user input country and year
#selected_country_year(country=country_userinput,year=year_userinput)

## NumPy (10)
# define a Numpy array from dataframe1(passengers) for year 2017
X = df1[['2017']].values
# find the maximum value in this array
max_value = np.max(X)
print('Maximum value of the array is',max_value)
#get the index for this Max value for each year 2017
max_index = np.argmax(X)
# max value for year 2015
#max_2015= max_index_col[0]
#print(max_2015)
# convert the whole dataframe 1 into numpy array
my_array = df1.to_numpy()
#print(my_array)
print(my_array.dtype) # this array type is object
# print country name with maximum passenger in 2017
print('the country with maximum passenger in 2017 is :', my_array[max_index][0])#noticing this line is world
# find the maximum value in this array
max_value = np.max(X)
print('Maximum value of the array is',max_value)
#get the index for this Max value for each year 2017
max_index = np.argmax(X)
# max value for year 2015
#max_2015= max_index_col[0]
#print(max_2015)
# convert the whole dataframe 1 into numpy array
my_array = df1.to_numpy()
#print(my_array)
print(my_array.dtype) # this array type is object
# print country name with maximum passenger in 2017
print('the country with maximum passenger in 2017 is :', my_array[max_index][0])#noticing this line is world



# define a Numpy array from dataframe2 (Population) for year 1960 and 2017
#1960 population
year_1960= data2.loc[data2["Year"] == 1960,['Country Name', 'Year', 'Population']]
#Sort 1960 population by descending
sorted_1960 = year_1960.sort_values(by='Population',ascending=False)

#2017 population
year_2017= data2.loc[data2["Year"] == 2017,['Country Name', 'Year', 'Population']]
#sort 2017 population by descending
sorted_2017 = year_2017.sort_values(by='Population',ascending=False)

#print first 5 rows to show top 5 countries by population
print(sorted_1960.head())
print(sorted_2017.head())
#after print head of sorted_1960 and sorted_2017, it is noticed that top 3 countries are the same

#to figure the growth for these 3 countries
top3_2017= sorted_2017.iloc[:3]
top3_1960= sorted_1960.iloc[:3]
top3_2017= sorted_2017.head(3)
top3_1960= sorted_1960.head(3)
print(top3_1960)

Pop2017 = top3_2017[['Population']].values
Pop1960 = top3_1960[['Population']].values
growth=(Pop2017-Pop1960)/Pop1960
print("top 3 countries population growth:",growth)

index = [1,2,3]
top3_1960 = top3_1960.reset_index(drop=True)
#top3_1960=top3_1960.set_index('Country Name')
top3_1960=top3_1960.rename(columns={'Population':'Population1960'})
del top3_1960['Year']
print(top3_1960)
top3_2017 = top3_2017.reset_index(drop=True)
#top3_2017=top3_2017.set_index('Country Name')
top3_2017=top3_2017.rename(columns={'Population':'Population2017'})
del top3_2017['Year']

print(top3_2017)

top3_1960vs2017= pd.merge(top3_1960, top3_2017)

# Add New Column 'Total_Growth' from population1960 to Population2017
top3_1960vs2017=top3_1960vs2017.assign(Total_Growth=lambda x: x.Population2017 / x.Population1960-1)
print(top3_1960vs2017)

# Add a New Column 'Average_Yearly_Growth' from 1960 to 2017 : 57 years
top3_1960vs2017=top3_1960vs2017.assign(Average_Yearly_Growth=lambda x: np.power(x.Population2017/x.Population1960,(1/57)))
print(top3_1960vs2017)

# Add a New Column 'Prediction2022' to predict these 3 country's population with Yearly_Growth from 2022 for 5 years
top3_1960vs2017_2022Predict=top3_1960vs2017.assign(Prediction2022=lambda x: x.Population2017 * x.Average_Yearly_Growth**5)
print(top3_1960vs2017_2022Predict)

##Looping(10)
# Based on the population dataframe, to find in which year India population will surpass China population
C2017=top3_1960vs2017_2022Predict.iloc[0,2]
C_AYG=top3_1960vs2017_2022Predict.iloc[0,4]
I2017=top3_1960vs2017_2022Predict.iloc[1,2]
I_AYG=top3_1960vs2017_2022Predict.iloc[1,4]


# initialize counter of year: i and for 1st year after 2017
i = 1
#China Population in year 2017+i
C_i=C2017*C_AYG**i
#India Population in year 2017+i
I_i=I2017*I_AYG**i

while I_i<C_i:
    i = i+1    # update counter of year
    C_i = C2017 * C_AYG ** i # update to next year China population
    I_i = I2017 * I_AYG ** i # update to next year India population

# print ' on which year, India population will surpass China population'
print("In the year", 2017+i, ", India population will surpass China population.")

##Dictionary or Lists(10)
# convert numpy array to list
ls = my_array.tolist()
# print
print("Type: ",type(ls))

# convert numpy array to dictionary
d = dict(enumerate(my_array.flatten(), 1))
# print dictionary
print("Type:",type(d))


##5. Visualise
##○ Seaborn, MatPlotlib (20)

#Seaborn plot
sns.set_theme(style="whitegrid")
#Plot1: top 5 countries by Population in 1960
g = sns.catplot(
    data=sorted_1960.head(), kind="bar",
    x="Country Name", y="Population",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Population in 1960")


#Plot2: top 5 countries by Population in 2017
g = sns.catplot(
    data=sorted_2017.head(), kind="bar",
    x="Country Name", y="Population",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Population in 2017")


#Plot 1960 and 2017 in same graph with MatPlotlib
x1 = sorted_1960['Country Name'].head()
y1 = sorted_1960['Population'].head()
x2 = sorted_2017['Country Name'].head()
y2 = sorted_2017['Population'].head()

fig, ax = plt.subplots()

# 1960 bar in red
bar1=plt.bar(x1,y1,tick_label=x1,width=0.4,color=['red'])
labels1 = ax.get_xticklabels()
plt.setp(labels1, rotation = 45, horizontalalignment = 'right')#Rotate the x label to better display

# 2017 bar in blue
bar2=plt.bar(x2,y2,tick_label=x2,width=0.2,color=['blue'])
labels2 = ax.get_xticklabels()
plt.setp(labels2, rotation = 45, horizontalalignment = 'right')#Rotate the x label to better display

plt.xlabel('Country Name')
plt.ylabel('Population')
plt.title("World Population top 5: 1960 vs 2017")


#4th plot: for top3_1960vs2017 and top3_1960vs2017_2022Predict
labels=top3_1960vs2017['Country Name']
y1960= top3_1960vs2017['Population1960']
y2017= top3_1960vs2017['Population2017']
#growth=top3_1960vs2017['Total_Growth']
average_yearly_growth=top3_1960vs2017_2022Predict['Average_Yearly_Growth']
prediction_2022=top3_1960vs2017_2022Predict['Prediction2022']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y1960, width, label='1960 Population')
rects2 = ax.bar(x + width/2, y2017, width, label='2017 Population')
rects3 = ax.bar(x + width, prediction_2022, width/2, label='2022 Population Prediction')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Population')
ax.set_title('Top3 countries population and growth')
ax.set_xticks(x, labels,fontsize=20)
ax.legend()

ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)
ax.bar_label(rects3, padding=1)

#define second y-axis to show 'Growth' that shares x-axis with current plot
ax2 = ax.twinx()
#add 'Growth' into same graph with green dots
ax2.plot(labels, average_yearly_growth, 'r*')
#add second y-axis label: Growth
ax2.set_ylabel('Average Yearly Growth', color='red', fontsize=20)

plt.show()
##6. Generate valuable insights
##○ 5 insights from the visualisation (20)
#from the plotted graphs:
#1. China, India and United States are always the top3 in both 1960 and 2017.
#2. 2 new countries are entered into top 5 in 2017: Indonesia and Pakistan,
#3. 2 countries are fall off from top 5 in 2017: Russian Federation and Japan.
#4. Population top3 countries are not changed in 1960 and 2017, but their growth are different:
#India has greated population growth among these 3 countries, China growth is 2nd, and Unites States growth is 3rd.
#5. Based on the population dataframe information, for prediction of 2022 population, China and India population are
# almost the same.
