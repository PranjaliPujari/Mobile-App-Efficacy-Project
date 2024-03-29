#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Importing the necessary libraries
import pandas as pd
import numpy as np
import pingouin as pg

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Extra
import warnings
warnings.filterwarnings(action="ignore")


# #### Part I - Data Preparation and Wrangling
# 
# ***Data Loading and Preparation***: 
# 
#   - Are there any missing values? Ideally, there shouldn't be any.
#   - Are there any variables which are of an incorrect datatype? For e.g. categorical columns are stored as integers in the data set. In case you encounter such issues, make sure you convert the columns to the correct datatypes.

# In[2]:


## Load the dataset
dataframe=pd.read_csv("data_app.csv")
dataframe.head()


# In[3]:


## Check the datatypes and missing values 
new_dataframe = dataframe.info()
new_dataframe
## Hint - You can evaluate both using a single method


# ##### Record your observations

# In[4]:


### Convert the columns which are of incorrect datatypes (if any) to the correct datatype
#All columns are in correct datatypes


# Observations:
#     
#     1]There are no missing value in dataframe.
#     2]All columns are in correct datatypes.

# #### Part II: Exploratory Data Analysis
# 
# ***Univariate Analysis I: Data summary of customers’ characteristics***
# 
#    - Use appropriate tables for the summary statistics and charts and describe the customers’ characteristics. You may restrict yourself to the univariate analysis of these variables:
# 
#           - Demographics: Gender, age, nationality
#           - Loyalty membership and tenure with the hotel chain (in months)

# In[5]:


### Before starting with Univariate Analysis, make sure to filter the dataframe to only include one period, either Post = 0 or Post = 1
### This way there won't be any repetitions in demographics data for the same customer

### WRITE CODE FOR FILTERING THE DATAFRAME
filtered_dataframe1 = dataframe.loc[(dataframe['Post']==0)]   #filtering dataframe for Post=0
filtered_dataframe1


# In[6]:


filtered_dataframe2 = dataframe.loc[(dataframe['Post']==1)]   #filtering dataframe for Post=1
filtered_dataframe2


# In[7]:


### Next let's start making plots to describe each customer characteristic using the above filtered dataframe


# In[8]:


### Hint - If you're unsure what plots/summary statistics to use, inspect the datatypes (categorical or numerical) for the demographic characteristics
### Next, revise your EDA I and EDA II modules to understand which charts are useful for a specific datatype


# In[9]:


### WRITE CODE FOR ANALYSING "GENDER" VARIABLE
customers_gender1 = sns.countplot(data=filtered_dataframe2, x="Gender", 
    color="green", order=filtered_dataframe2["Gender"].value_counts().index)
customers_gender1.set(xlabel="Customer's Gender", ylabel="Count")
plt.show()                                                          # Observation is after the app’s adoption i.e.post=1


# In[10]:


### WRITE CODE FOR ANALYSING "NATIONALITY" VARIABLE
nationalty1 = sns.histplot(filtered_dataframe2["Nationality"], color="grey", bins=5)  
# Observation is after the app’s adoption i.e.post=1


# In[11]:


### WRITE CODE FOR ANALYSING "AGE"
age1 = sns.histplot(filtered_dataframe2["Age"], color="yellow", bins=10)
# Observation is after the app’s adoption i.e.post=1


# In[12]:


### WRITE CODE FOR ANALYSING "LOYALTY"
loyalty1 = sns.histplot(filtered_dataframe2["Loyalty"], color="red", bins=10)
# Observation is after the app’s adoption i.e.post=1


# In[13]:


### WRITE CODE FOR ANALYSING "TENURE"
bin_start = 0
bin_end = 100
bin_width = 5
tenure1 = sns.histplot(filtered_dataframe2["Tenure"], color="orange",
     bins=range(bin_start, bin_end, bin_width))
# Observation is after the app’s adoption i.e.post=1


# #### Part II: Exploratory Data Analysis
# 
# ***Univariate Analysis II: Data summary of customers’ purchase behavior***
# 
#    - Use appropriate tables for the summary statistics and graphs and describe customers’ purchase behavior. You may restrict yourself to the univariate analysis of these variables::
# 
#           - Amount spent [Spending]
#           - Number of bookings [NumBookings]

# In[14]:


### For amount spent and number of bookings, we will get 2 values for the same customer
### One for Post = 0 and another for Post = 1
### You can analyze the data separately for Post = 0 and Post = 1 phases


# In[15]:


### WRITE CODE FOR FILTERING THE DATAFRAME
filtered_dataframe1 = dataframe.loc[(dataframe['Post']==0)]   #filtering dataframe for Post=0
filtered_dataframe1


# In[16]:


### WRITE CODE FOR FILTERING THE DATAFRAME
filtered_dataframe2 = dataframe.loc[(dataframe['Post']==1)]   #filtering dataframe for Post=1
filtered_dataframe2


# In[17]:


### WRITE CODE FOR ANALYSING "SPENDING"
Amount_spent_after_adoption = sns.histplot(filtered_dataframe1["Spending"], color="blue", bins=10)

# Observation before app adoption i.e.post=0


# In[18]:


### WRITE CODE FOR ANALYSING "SPENDING"
Amount_spent_after_adoption = sns.histplot(filtered_dataframe2["Spending"], color="blue", bins=10)
# Observation is after the app’s adoption i.e.post=1


# In[19]:


### WRITE CODE FOR ANALYSING "NUMBO0KINGS"
no_of_bookings_before_adoption = sns.histplot(filtered_dataframe1["NumBookings"], color="blue", bins=10)
# Observation before app adoption i.e. post=0


# In[20]:


### WRITE CODE FOR ANALYSING "NUMBO0KINGS"
no_of_bookings_after_adoption = sns.histplot(filtered_dataframe2["NumBookings"], color="blue", bins=10)
# Observation is after the app’s adoption i.e.post=1


# ***Multivariate Analysis:***
# 
#    - Construct relevant pivot tables, bar charts, and scatterplots to get a preliminary understanding of the relationship between customers’ characteristics and their purchase behavior. (Generally, bar charts are more informative in the case of categorical variables [e.g., the average of Spending broken up by Gender], while scatterplots convey more information in the case of numerical variables [e.g., Spending versus Age])

# In[21]:


### In this section, you are free to choose the variables you want to analyse and the number of analyses you want to perform.
### For example, you can peform the following analyses.

### Analyze the relationship Spending and Gender.
### Analyze the relationship between Spending and Age. 
### Analyze the relationship between Spending and Loyalty.
### .... and so on.

### As suggested before, you are free to choose the variables for analysis.
### Also, you're free to use the necessary tools (either pivot tables or visualizations or both) to perform the analyses
### However, make sure that you analyze the patterns for Spending and Number of Bookings against minimum 3 customer characteristics


# In[22]:


### Analyze the relationship between Spending and Gender.
relationship_Spending_and_Gender = sns.barplot(data=dataframe, x="Gender", y="Spending", color="gray", ci=False)
relationship_Spending_and_Gender.set(xlabel="Gender", ylabel="Spending")


# In[23]:


pivot_relationship_Spending_and_Gender = dataframe.pivot_table(index=['Gender'], values=['Spending'], aggfunc='sum')
pivot_relationship_Spending_and_Gender.head()  #pivot table of spending & gender


# In[24]:


### Analyze the relationship between Spending and Age. 
relationship_between_Spending_and_Age = sns.scatterplot(data=dataframe, x="Spending", y="Age", color="green")


# In[25]:


pivot_relationship_between_Spending_and_Age = dataframe.pivot_table(index=['Spending'], values=['Age'], aggfunc='sum')
pivot_relationship_between_Spending_and_Age.head()           #pivot table of spending & Age


# In[26]:


### Analyze the relationship between Spending and Loyalty.
relationship_between_Spending_and_Loyalty = sns.scatterplot(data=dataframe, x="Spending", y="Loyalty", color="green")


# In[27]:


pivot_relationship_between_Spending_and_Loyalty = dataframe.pivot_table(index=['Spending'], values=['Loyalty'], aggfunc='sum')
pivot_relationship_between_Spending_and_Loyalty.head()           #pivot table of spending & loyalty


# In[28]:


### Analyze the relationship between Spending and Nationality.
relationship_between_Spending_and_Nationality = sns.scatterplot(data=dataframe, x="Spending", y="Nationality", color="green")


# In[29]:


#pivot table of spending & nationality
pivot_relationship_between_Spending_and_Nationality = dataframe.pivot_table(index=['Spending'], values=['Nationality'], aggfunc='sum')
pivot_relationship_between_Spending_and_Nationality.head()


# In[30]:


### Analyze the relationship between Spending and Tenure.
relationship_between_Spending_and_Tenure = sns.scatterplot(data=dataframe, x="Spending", y="Tenure", color="green")


# In[31]:


#pivot table of spending & Tenure
pivot_relationship_between_Spending_and_Tenure = dataframe.pivot_table(index=['Spending'], values=['Tenure'], aggfunc='sum')
pivot_relationship_between_Spending_and_Tenure.head()


# In[32]:


### Analyze the relationship between Spending and NumBookings.
relationship_between_Spending_and_NumBookings = sns.scatterplot(data=dataframe, x="NumBookings", y="Spending", color="green")


# In[33]:


pivot_relationship_between_Spending_and_NumBookings = dataframe.pivot_table(index=['NumBookings'], values=['Spending'], aggfunc='sum')
pivot_relationship_between_Spending_and_NumBookings.head()


# In[34]:


### Analyze the relationship between Tenure and NumBookings.
relationship_relationship_between_Tenure_and_NumBookings = sns.scatterplot(data=dataframe, x="NumBookings", y="Tenure", color="green")


# In[35]:


### Analyze the relationship between Tenure and NumBookings.
pivot_relationship_relationship_between_Tenure_and_NumBookings = dataframe.pivot_table(index=['NumBookings'], values=['Tenure'], aggfunc='sum')
pivot_relationship_relationship_between_Tenure_and_NumBookings.head()


# In[36]:


### Analyze the relationship between Loyalty and NumBookings.
relationship_relationship_between_Loyalty_and_NumBookings = sns.scatterplot(data=dataframe, x="NumBookings", y="Loyalty", color="green")


# In[37]:


### Analyze the relationship between NumBookings and Loyalty.
pivot_relationship_relationship_between_NumBookings_and_Loyalty = dataframe.pivot_table(index=['NumBookings'], values=['Loyalty'], aggfunc='sum')
pivot_relationship_relationship_between_NumBookings_and_Loyalty.head()


# - Generate a table of the correlations of all numerical variables of the data set. 
#    
#    

# In[38]:


### Subset the dataframe to only include the numerical variables
numerical_variables=['CustomerID','Adopt','Age','Nationality','Loyalty','Tenure','Post','NumBookings','Spending']

### After that you can create a correlation matrix.
corr = dataframe.corr()
corr


# In[39]:


### If you want, you can also build a HeatMap, but it's optional.
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=0.5,annot=True)
plt.show()


#  - Determine whether there is a statistically significant difference between the average spending of men and women (at a 5% significance level)? Conduct an appropriate hypothesis test to determine whether there is a difference in means. Please construct a 95% confidence interval for the difference in means. You may assume independent samples and constant variance. [Note: The above test is to be conducted for the entire data set]

# In[40]:


### This task may seem intimidating at first.
### However, using the ttest method pingouin package which you learned in the "Designing Business Experiments" module, you should be able to get all the results directly.


# In[41]:


##Take male gender group in a separate DataFrame
Male = dataframe[dataframe['Gender']=="Male"]
Male.head()


# In[42]:


##Take female gender group in a separate DataFrame
Female = dataframe[dataframe['Gender']=="Female"]
Female.head()


# In[43]:


#### Calculate the average of the "Diff" column
Male['Spending'].mean() - Female['Spending'].mean()


# In[44]:


## Load the necessary libraries
from scipy.stats import ttest_1samp


# In[45]:


## Check the documentation
get_ipython().run_line_magic('pinfo', 'ttest_1samp')


# In[46]:


a=dataframe.query('Gender=="Male"')['Spending']
b=dataframe.query('Gender=="Female"')['Spending']


# In[47]:


import pingouin as pg
pg.ttest(a,b, paired=True)


# Observation: Since, p-value < 0.05, the difference is statistically significant at the 5% level.

# #### Part III: Statistical Analysis
# 
# ***After-Only design***
#    - To determine the treatment effect of customers’ adoption of the app on their spending with the hotel chain, construct a pivot table of average Spending broken up by Adopt and Post. What is the difference between the treatment and control groups’ spending in the Post =1 period? This is the treatment effect, assuming the experiment is of an After-Only design.

# In[48]:


### WRITE CODE FOR CONSTRUCTING A PIVOT TABLE
### Hint - Check the documentation of pivot table

pivot = dataframe.pivot_table(index=['Adopt', 'Post'], values=['Spending'], aggfunc='mean')
pivot   


# In[49]:


### Report the difference between the spending of treatment and control groups in Post = 1 period.
treatment_effect=pivot.loc[1,1]-pivot.loc[0,1]
print('Treatment effect is:', treatment_effect)


#   - Is the above treatment effect statistically significant? Perform the necessary hypothesis test and construct a 95% confidence interval for the difference. Take the level of significance as 0.05

# ***Before-After design***
#         
#         
# - Construct a new DataFrame, where for each customer, you have a new variable, which is the difference in spending between the Post = 1 and Post= 0 periods.
# 

# In[50]:


### Hint - Once again, you can use the ttest method in the pingouin package to perform this task quickly.
treatment_spending=dataframe.loc[(dataframe['Adopt']==1)& (dataframe['Post']==1),'Spending']
control_spending=dataframe.loc[(dataframe['Adopt']==0) & (dataframe['Post']==1), 'Spending']

t_result=pg.ttest(treatment_spending,control_spending, paired=False, alternative='two-sided',correction=True,confidence=0.95)

print("Test result: t-statistics = {:.3f}, p-value={:.3f}".format(t_result['T'].values[0], t_result['p-val'].values[0]))
print('95% confidence interval for the difference [{:.2f},{:.2f}]'.format(t_result['CI95%'][0][0], t_result['CI95%'][0][1]))


# Observations: Since, p-value < 0.05, the difference is statistically significant at the 5% level

# In[51]:


### This task can be slightly challenging and hence for performing this we have suggested the following approach
### Step 1 - Create 2 separate temporary dataframes, each filtered by Post = 1 and Post = 0 periods
data_post1=dataframe.loc[dataframe['Post']==1]
data_post0=dataframe.loc[dataframe['Post']==0]

### Step 2 - Merge these two dataframes using the pandas.merge() method based on the "CustomerID" and store this in a new Dataframe
merged_data=pd.merge(data_post1,data_post0,on='CustomerID', suffixes=('_post1','_post0'))

### Step 3 - After merging, you may encounter repeated columns (denoted by a suffix like "_x" & "_y" ). Identify which ones you want to keep
###, and which ones you want to drop. Make sure you keep the Spending columns for both Post = 1 and Post = 0 periods.
merged_data=merged_data[['CustomerID','Spending_post1','Spending_post0','Adopt_post1','Adopt_post0','Post_post1','Post_post0']]

### Step 4 - In the new merged Dataframe, create a new column which is the difference between the Spending in Post = 1 and Post = 0 for each customer
merged_data['Diff']=merged_data['Spending_post1']-merged_data['Spending_post0']

### For ease of use, you can name this column as "Diff". This will be useful in the final task of this section
print(merged_data.head())


# The above suggested approach is one of the many ways in which you can complete this task. Here are some helpful links for understanding how to use the `pandas.merge` method
# - [Documentation](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
# - [Youtube tutorial](https://www.youtube.com/watch?v=h4hOPGo4UVU)

# ---

# - Compute the average spending difference between those with Adopt = 1 and those with Adopt = 0 in both the Post = 1 and Post = 0 periods. Call these differences Difference1 and Difference0. Compute the difference between these two differences as Difference1 – Difference0. This is the treatment effect in the Before-After design.

# In[52]:


### To understand what why you're doing this, you can use the following image as a reference
### You have already learned this in the second session of the Designing Business Experiments module
### This is how you evaluate the treatment effect( or, Lift) in a Before-After Design


# ![BA Design.png](data:image/png;base64,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)

# In case the above image isn't opening please use this [link](https://images.upgrad.com/48d5bfcc-c3e5-4ef3-9c5a-460568c08480-Image2.png)
# 

# In[53]:


## Once you go through the image, the task is fairly straightforward to do. Here is one of the suggested approaches
## Calculate the average spending for Customers from control group (Adopt = 0) in the period Post = 0
filtered_dataframe1=dataframe.loc[(dataframe['Adopt']==0)& (dataframe['Post']==0),'Spending']

## Calculate the average spending for Customers from treamtent group (Adopt = 1) in the period Post = 0
filtered_dataframe2=dataframe.loc[(dataframe['Adopt']==1)& (dataframe['Post']==0),'Spending']

## Calculate the difference between the above two values. This will be Difference0 (or Treatment OEC(Before) - Control OEC (Before))
Difference0 = filtered_dataframe1.mean() - filtered_dataframe2.mean()
Difference0

## After the above, repeat the same steps for Post = 1 period
## Calculate the average spending for Customers from control group (Adopt = 0) in the period Post = 1
filtered_dataframe3=dataframe.loc[(dataframe['Adopt']==0)& (dataframe['Post']==1),'Spending']

## Calculate the average spending for Customers from treamtent group (Adopt = 1) in the period Post = 1
filtered_dataframe4=dataframe.loc[(dataframe['Adopt']==1)& (dataframe['Post']==1),'Spending']

## Calculate the difference between the above two values. This will be Difference1 (or Treatment OEC(After) - Control OEC (After))
Difference1 = filtered_dataframe3.mean() - filtered_dataframe4.mean()
Difference1

### Finally calculate the difference between these two values (Difference1 - Difference0) to obtain the treatment effect.
treatment_effect= Difference1-Difference0
print('Treatment effect is:', treatment_effect)


# - Is the above treatment effect statistically significant? Similar to the previous step, perform the necessary hypothesis test and construct a 95% confidence interval for the difference in differences. Take the level of significance as 0.05

# In[54]:


### Once again, you can peform this task using the pingouin package 
### In addition, you can use the "Diff" column from new DataFrame that you created in the first task of this section
### This will help in providing inputs to the ttest method from pingouin package.


# In[55]:


## Load the necessary libraries
from scipy.stats import ttest_1samp


# In[56]:


## Check the documentation
get_ipython().run_line_magic('pinfo', 'ttest_1samp')


# In[57]:


### Write the code for running a one-sample t-test on the "Diff" column
ttest_1samp(merged_data.Diff, popmean = 0)


# Observation: Since, p-value < 0.05, the difference is statistically significant at the 5% level.
