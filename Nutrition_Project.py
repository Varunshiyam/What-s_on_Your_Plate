#!/usr/bin/env python
# coding: utf-8

# # <center>  What's on your plate? </center>
# 
# **"An EDA of Nutrients in Different Food Categories"**
# 
# <center><img alt="Insight logo" src="https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8ZnJ1aXRzJTIwYW5kJTIwdmVnZXRhYmxlc3xlbnwwfHwwfHx8MA%3D%3D" align="center" hspace="10px" vspace="10px" width="900" height="1100" ></center>

# The project aims to analyze the nutrient values of food items in different categories. The dataset used is downloaded from Kaggle, and various Python libraries such as Pandas, Matplotlib, and Seaborn are used for data analysis and visualization. The project provides insights into the distribution of various nutrients such as protein, carbohydrates, fiber, etc., in different food categories. 

# <a id="table"> </a>
# <h1 style="background-color:red;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 20px 90px;">Table Of Content</h1>
# 
# 
# 
# * [1. Download the data](#1)
#     
# * [2. Data Preparation and Cleaning ](#2)
#   
# * [3. Exploratory Analysis and Visualization](#3)
# 
# * [4. Ask & Answer Questions ](#4)
# 
# * [4. Inferences and conclution](#5)
# 
# * [5.References and Future Work](#6)

#  ##     <a id="1"></a> <a id="table"></a>
# <h1 style="background-color:red;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Download the data</h1>  

# Download and install and import necessary packages.

# In[1]:


get_ipython().system('pip install opendatasets --upgrade --quiet')
import opendatasets as od
source_url = "https://www.kaggle.com/datasets/sathyakrishnan12/nutrition-datasets"
od.download(source_url)


# In[2]:


import os
os.listdir("nutrition-datasets")


# In[3]:


# import liabraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 100)
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[4]:


get_ipython().system('pip install jovian --upgrade --quiet')
project_name = "Nutrition-data-analysis"
import jovian
jovian.commit(project=project_name)


# ##   <a id="2"></a>  <h1 style="background-color:red;font-family:newtimeroman;font-size:200%;text-align:center;border-radius: 25px 50px;">Data Preparation and Cleaning </h1> 

# In[5]:


# read & load the dataset into pandas dataframe
df = pd.read_csv("nutrition-datasets/nutrition.csv")


# In[6]:


# printing the columns

for i in df.columns:
    print(i)


# In[7]:


df.info()


# **First lets do few thing befor analysis**
# 
# `This Project contains only some essentials vitamins & minerals`
# 
# * Dropping and selecting the columns/Rows which are not cover in this project
# * Check whether any null values present inside the dataframe or not.`
# * Rename the columns
# 

# In[8]:


df.columns


# Start with droping the columns which are note covered in this project

# In[9]:


#  Drop the columns by drop method
#df.drop(columns=['col1', 'col2', 'col4'], inplace=True)

# Or you  make a new data frame and add then select columns inside it.

df = df[[
    # food
    'name',
    'serving_size',
    'calories',
    # vitamins
     'vitamin_a','vitamin_b12',
       'vitamin_b6', 'vitamin_c', 'vitamin_d', 'vitamin_e','carotene_beta','thiamin',

    # Protein & minerals
     'copper', 'irom', 'magnesium',
       'phosphorous', 'potassium', 'zink', 'protein','calcium',
    # carbs , fats & sugars
    'carbohydrate', 'fiber', 'sugars',
       'fat','total_fat','cholesterol','sodium']].copy()


# we have to deal with name columns this cause some uncertainities. While doing EDA and visualization.

# In[10]:


df['name'] = df['name'].str.replace(',', '\n')

# split the columns
#df[["Food","Type"]] = df["name"].str.split(",", n =1, expand=True,)

# drop the original Name columns
#df = df.drop("name",axis=1)

# reindexing the columns 
#df = df.reindex(columns=['Food', 'Type'] + list(df.columns[:-2]))



# Check for the null value now

# In[11]:


df.columns[df.isna().any()]


# In[12]:


df.tail(1)


# <center><b>Abbreviations Used in this Project</b></center>
# 
# 
# 
# | Column name   | Description                                                                                               |
# |---------------|---------------------------------------------------------------------|
# | Food          | Name of the Food                                                                                          |
# | Svg_size_g    | Serving Size Gram                                                                                         |
# | Ttl_fat_      | Total Fat                                                                                                 |
# | Vit_e_mg      | Vitamin E (mg)                                                                                            |
# | Vit_D_IU      | Vitamin D (IU)                                                                                            |
# | Chol_(mg)     | Cholesterol (mg)                                                                                          |
# | Vit_b6_mg     | Vitamin B6 is a water-soluble vitamin that is part of the B-vitamin family.                                |
# | Vit_b12_mcg   | Vitamin B12 is a water-soluble vitamin that is part of the B-vitamin family.                                |
# | Vit_a_iu      | Vitamin A is a fat-soluble vitamin that is part of the vitamin A family.                                   |
# | B_carotene_mcg| Beta carotene is a fat-soluble vitamin that is part of the vitamin A family.                               |
# | Fe_mg         | Iron                                                                                                      |
# | Mg_mg         | Magnesium                                                                                                 |
# | P_mg          | Phosphorous                                                                                               |
# | K_mg          | Potassium                                                                                                 |
# | zn_mg         | Zinc                                                                                                      |
# | Protein_g     | Protein                                                                                                   |
# | Carbs_g       | Carbohydrate                                                                                              |
# | Fiber_g       | Fiber                                                                                                     |
# | Sugars_g      | Sugars                                                                                                    |
# | Fat_g         | Fat                                                                                                       |
# 

# In[13]:


df.rename(columns={"name":"Food","serving_size":"Svg_size_g",'total_fat':'Ttl_fat_g','cholesterol':'Chol_mg',"sodium":"Na_mg",'thiamin':'B1_mg','vitamin_a':'Vit_a_iu',"carotene_beta":'B_carotene_mcg',"vitamin_b12":'Vit_b12_mcg',
                   'vitamin_b6': 'Vit_b6_mg','vitamin_c':'Vit_c_mg','vitamin_d':'Vit_d_iu','vitamin_e':'Vit_e_mg','calcium':'Ca_mg','copper':'Cu_mg',
                   'irom':'Fe_mg','magnesium':'Mg_mg','phosphorous':'P_mg',"potassium":"K_mg","zink":'Zn_mg','protein':'Protein_g','carbohydrate':'Carbs_g','fiber':'Fiber_g','sugars':'Sugars_g','fat':'Fat_g'}, inplace = True)


# In[14]:


df.dtypes


# In[15]:



# dealing with miniscule amounts
cols_to_clean = list(df.columns[2:])
df[cols_to_clean] = df[cols_to_clean].replace({"mg": "", "g": "","mc":"","IU":""}, regex=True).astype(float)


# In[16]:


df.describe()


# One last time check for nan values

# In[17]:


df.columns[df.isna().any()]


# **Check for Duplicate**

# In[18]:


df.duplicated(subset=["Food"]).sum() 


# In[19]:


df.head()


# Now add a category column that contains a specific string that matches the `Food` name for the better analysis of this data

# In[20]:


# create a dictionary of food categories and their corresponding keywords
categories = {
    'Vegetables': ['broccoli', 'spinach', 'kale', 'carrots', 'tomatoes', 'bell peppers', 'cucumbers', 'zucchini', 'eggplant', 'cauliflower', 'cabbage', 'onions', 'sweet potatoes', 'green beans', 'peas', 'asparagus', 'Brussels sprouts', 'artichokes', 'beets', 'radishes'],
    'Meat': ['beef', 'pork', 'lamb', 'goat', 'chicken', 'turkey', 'duck','goose', 'rabbit', 'bison'],
    'Poultry': ['chicken', 'turkey', 'duck', 'quail', 'goose', 'pigeon', 'pheasant'],
    'Fat food': ['butter', 'cheese', 'lard', 'margarine', 'shortening'],
    'Dairy Products': ['milk', 'yogurt', 'cheese', 'cream', 'butter', 'ice cream', 'sour cream', 'cottage cheese'],
    'Drinks and alcohol': ['water', 'coffee', 'tea', 'soda', 'juice', 'beer', 'wine', 'liquor'],
    'Dessert Sweets': ['cake', 'pie', 'chocolate', 'candy', 'cookies', 'ice cream'],
    'Seeds and Nuts': ['almonds', 'cashews', 'peanuts', 'pistachios', 'walnuts', 'sesame seeds', 'flax seeds', 'chia seeds']
}

# create a dictionary of food categories 

df['Category'] = ''

# iterate through each food category and assign the corresponding category to the food items
for category, keywords in categories.items():
    df.loc[df['Food'].str.contains('|'.join(keywords), case=False), 'Category'] = category


# In[21]:


# Now delete the remaining rows which are not in the categories.
df['Category'].replace('', np.nan, inplace=True)

df.dropna(subset=["Category"], inplace = True)
df.reset_index(drop=True , inplace = True)


# In[22]:


df["Category"].value_counts()


# ##      <a id="3"></a>  <h1 style="background-color:red;font-family:newtimeroman;font-size:200%;text-align:center;border-radius: 25px 50px;">Exploratory Analysis and Visualization </h1> 

# In[23]:


fig, ax = plt.subplots(figsize = (20,10), dpi = 100)
sns.set(style="whitegrid")
Category_count = df['Category'].value_counts()
ax = sns.barplot(x=Category_count.index,y=Category_count.values ,palette='mako');

# set x & y label
ax.set_ylabel('Food Counts', fontsize = 20)
ax.set_xlabel("Food Category", fontsize = 20)

# grid color
ax.grid(color='#6495ed', linewidth=1, axis='y', alpha=.3)

# set title
ax.set_title("Distribution of Food Categories", fontsize = 25)

plt.show()


# Majority of food items belong to the `Meat` category, followed by `Drinks and alcohal` and `Dairy Products`. The `Dessert Sweets` and `Poultry` categories also have a significant number of food items.
# 
# On the other hand, the `Vegetables`, `Fat food`, and `Seeds and Nuts` categories have a relatively small number of food items in the dataset.

# In[24]:




# create a new dataframe with only the necessary columns
Calories_df = df[['Category', 'calories']]

# create a histogram for each food category
plt.rc('font', family='serif', size=12)
fig, ax = plt.subplots(figsize = (10,5), dpi = 100)
ax = sns.histplot(data=df, x='calories', hue='Category', multiple='stack')

# set title & labels
ax.set_title("Distribution of Calories", fontsize = 20)
ax.set_xlabel("Calories Per 100 gram Serving")
ax.set_ylabel(" Value Counts")
ax.set_xticks(np.arange(0,1000,100))

# set x-tick labels to show "kcal"
x_ticks = plt.xticks()[0]
plt.xticks(x_ticks, [f"{int(x)} kcal" for x in x_ticks])

plt.show()


# **Insight**
# 
# Based on the above Histograph plot of calorie values for each food category, some observation are :-
# 
# * The majority of the Meat item fall in the range of 150 to 300 kcal per 100 g serving.
# * Poultry items are mostly in the range of 150-200 kcal per 100g serving with some outliers above 300 kcal
# * Dairy Products have a wide range of calorie counts , with many items falling in the range
# * Suprisingly desserts sweets are in the wide range 200 to 500 kcal per 100 gram serving
# * Fat food items have a high range of calories value,with most numbers are falling above 500 or some are over 800 kcal per 100 gram servings.
# * Vegetables have a low range of calories values,with most item falling below 100 kcal .
# 
# Overall, it can be concluded that different food categories have varying ranges of calories values,with some category having a wide range some have a naroow range. It is important to be aware of the calorie content of foods when making dietary choices

# In[25]:



# create a scatter plot of protein vs. carbohydrates
plt.rc('font', family='sans-serif', size=12)
plt.subplots(figsize = (10,8), dpi = 100)
sns.scatterplot(x='Protein_g',hue = 'Category', y='Carbs_g', data=df)

# set the title and x and y axis labels

plt.title('Protein vs. Carbohydrates Scatter Plot')
plt.xlabel('Protein (g)')
plt.ylabel('Carbohydrates (g)')

#

# show the plot
plt.show()


# **Insight**
# 
# Based on the above scatter plot , we can see that
# - The meat category has a high concentration of protein and relatievely low carbs, with some outliers that have high carbs.
# - Vegetables has a wide range of carbs, with a maximum of aroung 40-80 g , and a relatievely low concentration.
# - The poultry category has a relatively low concentration of carbs,with most of the data points having below 40 and a high no of protein(g) per 100 gram serving
# - The drinks and alcohal category are most in the range of below 20 carbs and relatievely low concentration of Protein .But some drinks has protein obviously they are juices.
# - The dairy products category has a wide range of nutrient value , with some outliers having high protein and some having high carbs
# 
#     `Overall, scatetr plot shows that there is an wide range of nutrient value accross different food category.The meat category is high in protein , whereas as vegetables are high in carbs, and poultry is relatively low in carbs and high in protein.While the dairy category has a wide range of nutrient value`

# In[26]:


# filter the dataframe for Meat and Vegetables categories
meat_vegetables_df = df[df["Category"].isin(["Dairy Products", "Meat"])]

# group the filtered dataframe by Category and calculate mean values of calories, protein, and fat
grouped_df = meat_vegetables_df.groupby("Category")[["Protein_g", "Fat_g"]].mean()

grouped_df.plot(kind="bar", figsize=(12,8))
plt.title("Meat vs Dairy Products", fontsize=16)
plt.yticks(np.arange(0,30,5))
plt.ylabel("Nutrition Values", fontsize=14)
plt.xlabel("Category", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.legend(fontsize=12)
plt.show()


# **Insight**
# 
# Based on tha analysis of the Meat and dairy products categories.it can observed that meat are slightly higher protein compared to the dairy products .On the other hand , dairy products are slightly higher in fat compared to meat products
# 
# This suggest that individuall looking for high protein diet may prefer Meat products.
# While those looking to substitute protein with other food option can consider dairy products.
# 
# `However, it is important dairy prducts are also high in fat content consider this data 100gram of per seving`
# 
# individuals should be mindful of their overall nutrient intake and consider low-fat dairy options if necessary.

# In[27]:


df.columns


# In[28]:



# create a correlation matrix of the nutrients
corr_matrix = df[['calories', 'Fat_g', 'Protein_g', 'Carbs_g', 'Fiber_g', 'Sugars_g']].corr()

# plot the correlation matrix as a heat map using seaborn
plt.figure(figsize=(15,8))
ax = sns.heatmap(corr_matrix, cmap='coolwarm')

plt.show()


# **Insight**
# 
# - The correlation heatmap shows that there is a negative correlation (-0.4) dark blue color between carbohydrates and food.As noted in the scatter plot of protein vs. carbohydrates, higher protein foods tend to have lower levels of carbohydrates.
# 
# - Similarly, there is a weak negative correlation between protein and fat (-0.2), indicating that foods high in protein tend to be lower in fat.
# 
# - On the other hand, there is a strong positive correlation between calories and fat (0.8), which means that high-calorie foods tend to be high in fat. 
# 
# - Positive correlation between carbs and sugars (0.6), suggesting that foods high in carbs often contain high levels of sugars.

# ##    <a id="4"></a> <h1 style="background-color:red;font-family:newtimeroman;font-size:200%;text-align:center;border-radius: 25px 50px;"> Ask & Answer Questions </h1> 

# ### **Q. what are the top categories from high to low protein?"**

# In[29]:


grouped_by = df.groupby("Category")["Protein_g"].mean()

sorted_series = grouped_by.sort_values(ascending=False)
plt.subplots(figsize = (15,8), dpi = 100)
plt.bar(sorted_series.index,sorted_series.values,)
plt.xlabel("Category")
plt.ylabel("Protein Content")
plt.title("Top Categories by protein content", size=15)

plt.show()


# * Meat, seeds and nuts, and poultry are the top three categories with the highest protein content. This suggests that these foods are good sources of protein for individuals who are looking to increase their protein intake.
# 
# * It is interesting to note that drinks and alcohol are also among the top categories with high protein content. This could be due to the fact that some drinks, such as protein shakes and smoothies, are specifically designed to provide high amounts of protein.
# 
# * Dessert sweets, vegetables, and fat food are among the categories with the lowest protein content. This suggests that individuals who are looking to increase their protein intake should focus on incorporating more foods from the top categories into their diet.

# ### **Q: What are the top 5 food items with the highest protein content?**

# In[30]:


high_protein_df = df[df["Protein_g"] > 20]

top_5 = high_protein_df.sort_values("Protein_g",ascending=False).head(5)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_5['Food'], top_5['Protein_g'], color='blue')

# set the axis labels and title
plt.xlabel("Protein (g)")
plt.ylabel("Food")
plt.title("Top 10 Foods by Protein Content")

plt.show()


# **Insights** \
# The bar plot represents the top 5 food items with the highest protein content. It shows that snacks plain pork skins have the highest protein content, followed by Snacks barbecue flavor pork skins, protein supplement powder muscle milk-based, protein supplement powder muscle milk-based, and cheese low sodium parmesan. It is interesting to note that all of these items belong to the snacks and supplement category. This suggests that people who want to increase their protein intake tend to rely on supplements and snacks rather than whole foods. It is also worth noting that these items are high in protein but may not necessarily be healthy due to their high fat content. Therefore, it is important to balance protein intake with a variety of other healthy foods.

# ### **Q Which food categories are the highest sources of Minerals ?**

# In[31]:


df.columns


# In[32]:


minerals_df = df[["Category","Zn_mg","Mg_mg","P_mg","Fe_mg"]]

grouped_df = minerals_df.groupby("Category").mean()

grouped_df.plot(kind="bar", figsize=(12,8))
plt.xticks(rotation=45)

plt.show()


# This shows that seeds and nuts are the highest sources of both potassium and magnesium. Meat is also a good source of potassium, but not as high as seeds and nuts. Dairy products, especially milk and cheese, are good sources of Potasssium but not as high in potassium as seeds and nuts

# ### **Q. what are the top categories from high to low Calories?"**

# In[33]:


grouped_by = df.groupby("Category")["calories"].mean()

sorted_series = grouped_by.sort_values(ascending=False)
plt.subplots(figsize = (15,8), dpi = 100)
plt.bar(sorted_series.index,sorted_series.values,)
plt.xlabel("Category")
plt.ylabel("Carories Content")
plt.title("Top Categories by Calories", size=15)

plt.show()


# Insights based on the bar plot:
# 
# * The highest calorie category is seeds and nuts, which may come as a surprise to some people who think of these foods as healthy snacks.
# * The fact that fat foods are the second-highest calorie category may not be surprising to many people, as these foods are often associated with high calorie content.
# * Dessert sweets being the third-highest calorie category is also not surprising, as these foods are often high in sugar and fat.
# * Dairy products are relatively high in calories compared to other food categories, which may be unexpected for some people who view dairy as a healthy source of calcium.
# * Meat and poultry are also high in calories, which may be expected for some people who associate these foods with protein and energy.
# * Finally, drinks and alcohol are the lowest calorie category, which may be expected for some people who think of liquid calories as less filling or satisfying than solid food. However, it's worth noting that some alcoholic beverages can be high in calories, especially if they contain added sugar or mixers.

# ### **Q: Create a Dataframe Category with most fiber**

# In[34]:


top_fiber = df.groupby('Category')['Fiber_g'].mean().nlargest(10)

top_fiber_df = top_fiber.reset_index(name='Mean Fiber (g)').head(10)
top_fiber_df


# In[35]:


jovian.commit()


# ##   <a id="5"></a> <h1 style="background-color:red;font-family:newtimeroman;font-size:200%;text-align:center;border-radius: 25px 50px;">  Inferences and conclution </h1> 

# 
# The main aim of the project is to analyze the nutritional value of food items across different categories. Through our analysis, we found that the meat category has a high concentration of protein, whereas seeds and nuts are rich in fiber. Categories such as fat foods and desserts are high in calories. Our conclusions are based on analyzing various food categories. 
# 
# we can conclude that different food categories have different nutrient compositions, and it's important to consider these differences when planning a balanced diet. By identifying the categories that are rich in specific nutrients, we can also make informed choices about the types of food we consume to meet our nutritional needs. Overall, this project provides valuable insights into the nutrient composition of different food categories and can be useful for both individuals and professionals in the food and health industries.
# 
# 

# ##    <a id="6"></a> <h1 style="background-color:red;font-family:newtimeroman;font-size:200%;text-align:center;border-radius: 25px 50px;">  References and Future Work </h1>  

# **References**
# 
# https://matplotlib.org/
# 
# https://pandas.pydata.org/
# 
# https://stackoverflow.com/
# 
# https://www.kaggle.com/
# 
# Youtube videos :-
# 
# Rob Mulla - [Exploratory Data Analysis with Pandas Python 2023](https://youtu.be/xi0vhXFPegw) \
# Jovian - [Build an Exploratory Data Analysis Project from Scratch with Python, Numpy, and Pandas](https://www.youtube.com/live/kLDTbavcmd0?feature=share)
# 
# 
# 
# Datasets Source \
# [kaggle](https://www.kaggle.com/datasets/sathyakrishnan12/nutrition-datasets) 
# 
# 
# 
# **Future work:** \
# In the future, machine learning techniques could be applied to the nutrition data to create personalized dietary recommendations or to predict nutrient deficiencies based on dietary habits. 
# 
# 
# 
