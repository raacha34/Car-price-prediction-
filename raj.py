import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/13174/Downloads/Car_features.csv')  
    return data

df = load_data()

# Streamlit app layout
st.title('Car Features Analysis')

# Displaying the DataFrame
if st.checkbox('Show DataFrame'):
    st.dataframe(df)

# Analysis and Visualization of Transmission Type Over the Years
st.title('Preferred Transmission Type Over The Years')

# Grouping and sorting data
transmission = df.groupby(['Year', 'Transmission Type'])['Make'].agg(['count']).sort_values(by=['Year', 'count'], ascending=True).reset_index()

# Creating the plotly figure
fig = px.line(transmission, x='Year', y='count', color='Transmission Type', title='Preferred Transmission Type Over The Years', template='plotly_dark')
fig.update_xaxes(tickmode='linear', type='category')

# Displaying the plot
st.plotly_chart(fig)

# Subset of Cars Dataset
st.title('Subset of Cars Dataset')
subset_size = st.slider('Number of Rows to Display', 1, len(df), value=25) 
st.dataframe(df.head(subset_size))
