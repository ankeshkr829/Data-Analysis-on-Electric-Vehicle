#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv(r"C:\Users\ankes\OneDrive\Desktop\data2\Vehicle_dataset.csv")


# In[3]:


print(df.head())


# # Task 1- (Univariate Analysis and Bivariate Analysis )

# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna(inplace=True)
df.shape


# # Univariate Analysis

# In[7]:


#Summary statistics
print(df.describe())


# In[8]:


# Create histograms for numerical features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# # Bivariate Analysis

# In[9]:


# Scatter plots
for col1 in numerical_columns:
    for col2 in numerical_columns:
        if col1 != col2:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col1, y=col2)
            plt.title(f'Scatter Plot: {col1} vs. {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()


# In[10]:


# Exclude non-numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]

# Correlation matrix
correlation_matrix = df_numeric.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# # Task 2- Choropleth map using plotly.express

# In[11]:


import plotly.express as px

vehicle_type_count = df.groupby('Electric Vehicle Type')['VIN (1-10)'].count().reset_index()
vehicle_type_count.columns = ['Electric Vehicle Type', 'Count']  

fig = px.pie(vehicle_type_count, 
             names='Electric Vehicle Type', 
             values='Count', 
             title='Distribution of Electric Vehicles by Type',
             labels={'Electric Vehicle Type': 'EV Type', 'Count': 'Number of Vehicles'},
             hole=0.3, 
             color_discrete_sequence=px.colors.sequential.Plasma) 

fig.update_traces(textinfo='percent+label')  
fig.update_layout(title_font_size=20,  
                  margin=dict(l=40, r=40, t=40, b=40))  


# In[12]:


df['Longitude'] = df['Vehicle Location'].apply(lambda loc: float(loc.split()[1][1:]))
df['Latitude'] = df['Vehicle Location'].apply(lambda loc: float(loc.split()[2][:-1]))

location_counts = df.groupby(['Latitude', 'Longitude', 'Postal Code', 'County', "State"]).size().reset_index(name='EV Count')
 
    


# In[13]:


fig_scatter_map = px.scatter_mapbox(location_counts,
                                    lat='Latitude',
                                    lon='Longitude',
                                    color='EV Count',
                                    size='EV Count',
                                    mapbox_style='carto-positron',
                                    zoom=3,
                                    center={'lat': 37.0902, 'lon': -95.7129},
                                    title='Scatter Map of Electric Vehicle Locations')

fig_scatter_map.show()


# # Task 3-Racing Bar Plot using plotly.express

# In[14]:


import bar_chart_race as bcr


# In[15]:


data = df.groupby(['Make', 'Model Year']).size().reset_index(name='Number_of_Vehicles')


# In[16]:


import plotly.express as px
fig = px.bar(data,
             y='Make',
             x='Number_of_Vehicles',
             animation_frame='Model Year',
             orientation='h',
             title='EV Makes and their Count Over the Years',
             labels={'Number_of_Vehicles': 'Number of EV Vehicles'},
             range_x=[0, 3000],
             color='Make', 
             color_discrete_map={
                 'Tesla': 'red',
                 'Toyota': 'blue',
                 'Ford': 'green',
             }
             )

# Find the year with the highest number of vehicles
top_year = data['Model Year'].max()

# Check if the year exists and add the annotation
if (df['Model Year'] == top_year).any():
    make_2023 = df.loc[df['Model Year'] == top_year, 'Make'].iloc[0]
    fig.add_annotation(x=2500, y=make_2023,
                       text=f"Most EVs: {top_year}",
                       showarrow=False,
                       font_size=18)
else:
    print("Year 2023 not found in data")

# Customize the layout
fig.update_layout(
    xaxis=dict(showgrid=True, gridcolor='LightGray'),
    yaxis_title='EV Makes',
    xaxis_title='Number of EV Vehicles',
    showlegend=False,
    title_x=0.5,
    title_font=dict(size=20),
    margin=dict(l=90, r=90, t=90, b=90),
    width=800,
    height=600
    )

# Customize trace appearance
fig.update_traces(texttemplate='%{x}',
                  textposition='outside',
                  textfont_size=20)

# Show the plot
fig.show()


# In[ ]:




