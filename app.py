import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Importer les bibliothèques nécessaires
import pandas as pd
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# Définir les chemins des fichiers
file1 = '/Users/philippebeliveau/Desktop/TEC/temperaturemiso.csv'
file2 = '/Users/philippebeliveau/Desktop/TEC/temperaturenyiso.csv'
file3 = '/Users/philippebeliveau/Desktop/TEC/temperaturepjm.csv'

# Lire les fichiers csv
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Concaténer les trois dataframes
df = pd.concat([df1, df2, df3])

def determine_season(date):
    if (3 <= date.month <= 5) or (date.month == 2 and date.day >= 20):
        return "Spring"
    elif (6 <= date.month <= 8):
        return "Summer"
    elif (9 <= date.month <= 11):
        return "Autumn"
    else:
        return "Winter"

df['iso_id'] = df['iso_id'].map({2: 'NYISO', 3: 'MISO', 4: 'PJM'})
df['season'] = pd.to_datetime(df['load_date']).apply(determine_season)
df['hour'] = pd.to_datetime(df['load_date']).dt.hour

# Title and header
st.title("TEC Energy Data Analyst Test")
st.markdown("### By Philippe Beliveau")
st.markdown("#### Date: 23 September 2024")

# Display region analysis
st.header("Analysis of the Regions")

# MISO Analysis
st.subheader("MISO")
st.image("/Users/philippebeliveau/Desktop/TEC/MISO.jpg", caption="MISO Region", width=300)
st.markdown(
    """
    MISO operates the transmission system and a centrally dispatched market in portions of 15 states in the Midwest and the South, extending from Michigan and Indiana to Montana and from the Canadian border to the southern extremes of Louisiana and Mississippi. The system is operated from three control centers: Carmel, Indiana; Eagan, Minnesota; and Little Rock, Arkansas. MISO also serves as the reliability coordinator for additional systems outside of its market area, primarily to the north and northwest of the market footprint.

    [More about MISO](https://www.ferc.gov/industries-data/electric/electric-power-markets/miso)
    """
)

# NYISO Analysis
st.subheader("NYISO")
st.image("/Users/philippebeliveau/Desktop/TEC/NYISO.jpg", caption="NYISO Region", width=300)
st.markdown(
    """
    The creation of the New York Independent System Operator (NYISO) was authorized by FERC in 1998 and launched on Dec. 1, 1999. The NYISO footprint covers the entire state of New York. NYISO is responsible for operating wholesale power markets that trade electricity, capacity, transmission congestion contracts, and related products, in addition to administering auctions for the sale of capacity. NYISO operates New York’s high-voltage transmission network and performs long-term planning.

    What are the main source of energy in NYISO?
    *Sneak peak on the source of variability*
    [More about NYISO](https://www.ferc.gov/industries-data/electric/electric-power-markets/nyiso)
    """
)

# PJM Analysis
st.subheader("PJM")
st.image("/Users/philippebeliveau/Desktop/TEC/PJM.jpg", caption="PJM Region", width=300)

st.markdown(
    """
    The PJM Interconnection operates a competitive wholesale electricity market and manages the reliability of its transmission grid. PJM provides open access to the transmission and performs long-term planning. In managing the grid, PJM centrally dispatches generation and coordinates the movement of wholesale electricity in all or part of 13 states (Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia and West Virginia) and the District of Columbia. PJM’s markets include energy (day-ahead and real-time), capacity, and ancillary services.

    [More about PJM](https://www.ferc.gov/industries-data/electric/electric-power-markets/pjm)
    """
)

# ****************************************************

# """
# Scatter plot of temperature against load forecast
# """

st.header("Visualisation of Temperature and Load Forecast per ISO")

# Define the analysis text for each ISO ID
analysis_text = {
    'MISO': "MISO shows a strong positive correlation of 0.90 between temperature and forecasted load. This indicates that as temperature increases, there is a significant increase in the forecasted load. The upward trend is especially noticeable beyond 20°C, suggesting a high dependency on temperature for load forecasting.",
    'NYISO': "NYISO exhibits a correlation of 0.77, indicating a positive but more variable relationship between temperature and load forecast. The data points spread more broadly, suggesting that other factors may also significantly influence load forecasts in New York.",
    'PJM': "PJM, similar to MISO, demonstrates a strong correlation of 0.90, showing that temperature increases reliably predict higher load forecasts. The data points are densely clustered, indicating a strong consistency in the relationship across the region."
}

# Get unique iso_ids
iso_ids = df['iso_id'].unique()

# Create a dropdown to select an iso_id
selected_iso_id = st.selectbox("Select ISO ID", iso_ids)

# Filter the DataFrame for the selected iso_id
filtered_df = df[df['iso_id'] == selected_iso_id]

# Display analysis text for the selected ISO ID
if selected_iso_id in analysis_text:
    st.write(analysis_text[selected_iso_id])

# Create the scatter plot
if not filtered_df.empty:
    # Calculate correlation coefficient
    correlation = filtered_df['temperature_celsius'].corr(filtered_df['load_forecast'])
    
    # Create the scatter plot
    fig = px.scatter(filtered_df, x="temperature_celsius", y="load_forecast",
                     title=f"Relationship Between Temperature and Forecasted Load for {selected_iso_id} (Correlation: {correlation:.2f})",
                     labels={"temperature_celsius": "Temperature (°C)", "load_forecast": "Load Forecast"})
    
    # Display the figure
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected ISO ID.")


# ****************************************************

# """
# Time series over the day (Dual axis)
# """

# Function to calculate means for a given region and season
def calculate_means(df):
    return df.groupby(['iso_id', 'hour']).mean(numeric_only=True).reset_index()

# Prepare the data (assuming df is your DataFrame)
means = calculate_means(df)

# Create a subplot with shared x-axis but different y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add temperature and load forecast traces for all regions and seasons
for iso_id in means['iso_id'].unique():
    iso_season_data = means[(means['iso_id'] == iso_id)]
    # Temperature trace
    fig.add_trace(go.Scatter(
        x=iso_season_data['hour'],
        y=iso_season_data['temperature_celsius'],
        mode='lines+markers',
        name=f'{iso_id} Temperature',
        line=dict(width=2),
        marker=dict(size=6),
        visible='legendonly'  # Initially set to legend only
    ), secondary_y=False)

    # Load forecast trace
    fig.add_trace(go.Scatter(
        x=iso_season_data['hour'],
        y=iso_season_data['load_forecast'],
        mode='lines+markers',
        name=f'{iso_id} Load Forecast',
        line=dict(dash='dash', width=2),
        marker=dict(size=6),
        visible='legendonly'  # Initially set to legend only
    ), secondary_y=True)

# Update layout
fig.update_layout(
    title="Temperature and Load Forecast",
    xaxis_title="Hour of Day",
    template='plotly_white'
)

# Set y-axis titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
fig.update_yaxes(title_text="Load Forecast", secondary_y=True)

# Create dropdown menus for filtering by ISO and season
region_buttons = [
    dict(label=iso_id, method='update', args=[
        {'visible': [iso_id in trace.name for trace in fig.data]},  # Show only the selected ISO
        {'title': f"Temperature and Load Forecast for {iso_id}"}
    ]) for iso_id in means['iso_id'].unique()
]

# Update the layout with dropdown menus
fig.update_layout(
    annotations=[
        dict(text="Select Region", x=0.1, xref="paper", y=1.1, yref="paper", showarrow=False),
    ],
    updatemenus=[
        {
            'buttons': region_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }
    ]
)

# Display the figure in Streamlit
st.plotly_chart(fig)

# ****************************************************

# """
# Time series over the whole time period (Dual axis)
# """

# Function to calculate means for a given region and season
def calculate_means(df):
    return df.groupby(['iso_id', 'load_date']).mean(numeric_only=True).reset_index()

# Prepare the data (assuming df is your DataFrame)
means = calculate_means(df)

# Create a subplot with shared x-axis but different y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add temperature and load forecast traces for all regions and seasons
for iso_id in means['iso_id'].unique():
    iso_season_data = means[(means['iso_id'] == iso_id)]

    # Temperature trace
    fig.add_trace(go.Scatter(
        x=iso_season_data['load_date'],
        y=iso_season_data['temperature_celsius'],
        mode='lines+markers',
        name=f'{iso_id} Temperature',
        line=dict(width=2),
        marker=dict(size=6),
        visible='legendonly'  # Initially set to legend only
    ), secondary_y=False)

    # Load forecast trace
    fig.add_trace(go.Scatter(
        x=iso_season_data['load_date'],
        y=iso_season_data['load_forecast'],
        mode='lines+markers',
        name=f'{iso_id} Load Forecast',
        line=dict(dash='dash', width=2),
        marker=dict(size=6),
        visible='legendonly'  # Initially set to legend only
    ), secondary_y=True)

# Update layout
fig.update_layout(
    title="Temperature and Load Forecast",
    xaxis_title="Date",
    template='plotly_white'
)

# Set y-axis titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
fig.update_yaxes(title_text="Load Forecast", secondary_y=True)

# Create dropdown menus for filtering by ISO and season
region_buttons = [
    dict(label=iso_id, method='update', args=[
        {'visible': [iso_id in trace.name for trace in fig.data]},  # Show only the selected ISO
        {'title': f"Temperature and Load Forecast for {iso_id}"}
    ]) for iso_id in means['iso_id'].unique()
]
# Update the layout with dropdown menus
fig.update_layout(
    annotations=[
        dict(text="Select Region", x=0.1, xref="paper", y=1.1, yref="paper", showarrow=False),
    ],
    updatemenus=[
        {
            'buttons': region_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        },
    ]
)

# Display the figure in Streamlit
st.plotly_chart(fig)

# ****************************************************

# """
# Bar plot with error bars
# """

# Define temperature ranges
temperature_bins = [-float('inf'), 0, 10, 20, 30, float('inf')]
temperature_labels = ['Below 0°C', '0-10°C', '10-20°C', '20-30°C', 'Above 30°C']

# Get unique iso_ids
iso_ids = df['iso_id'].unique()

# Create a dropdown to select an iso_id with a unique key
selected_iso_id = st.selectbox("Select ISO ID", iso_ids, key='iso_selectbox')

# Filter the DataFrame for the selected iso_id
filtered_df = df[df['iso_id'] == selected_iso_id]

# Create temperature bins
filtered_df['temperature_range'] = pd.cut(filtered_df['temperature_celsius'], bins=temperature_bins, labels=temperature_labels)

# Calculate average load forecast and standard deviation for error bars
average_load = filtered_df.groupby('temperature_range')['load_forecast'].agg(['mean', 'std']).reset_index()
average_load.columns = ['Temperature Range', 'Average Load Forecast', 'Standard Deviation']

# Create the bar chart with error bars
fig = px.bar(average_load,
             x='Temperature Range',
             y='Average Load Forecast',
             error_y='Standard Deviation',
             title=f'Average Load Forecast for Different Temperature Ranges in {selected_iso_id}',
             labels={'Average Load Forecast': 'Average Load Forecast', 'Temperature Range': 'Temperature Range'},
             color='Temperature Range')

# Display the bar chart in Streamlit
st.plotly_chart(fig)

# ****************************************************

# """
# Bubble Chart
# """
import streamlit as st
import pandas as pd
import plotly.express as px

# Load and prepare your data
df['load_date'] = pd.to_datetime(df['load_date'], errors='coerce')
df['forecast_difference'] = df['load_forecast'] - df['load_forecast'].mean()

iso_ids = df['iso_id'].unique()
selected_iso_id = st.selectbox("Select ISO ID", iso_ids, key=f'iso_selectbox_{selected_iso_id}')

# Filter the DataFrame for the selected iso_id
filtered_df = df[df['iso_id'] == selected_iso_id]

# Prepare the data: calculate yesterday's load forecast
filtered_df['load_forecast'] = filtered_df['load_forecast']# .shift(1)

# Calculate average load forecast for the size of the bubble
average_forecast = filtered_df['load_forecast'].mean()
filtered_df['forecast_difference'] = filtered_df['load_forecast'] - average_forecast

# Filter out rows with NaN or negative values for the size
filtered_df = filtered_df.dropna(subset=['temperature_celsius', 'load_forecast', 'forecast_difference'])
filtered_df = filtered_df[filtered_df['forecast_difference'] >= 0]  # Keep only non-negative differences

# Display analysis comments based on the selected ISO ID
if selected_iso_id == "MISO":
    st.markdown("""
    ### MISO Analysis
    - **Trend**: There is a clear upward trend indicating that higher temperatures are associated with higher load forecasts.
    - **Color Coding**: The transition from cooler to warmer colors through the day shows the impact of daily temperature variations.
    """)
elif selected_iso_id == "NYISO":
    st.markdown("""
    ### NYISO Analysis
    - **Variability**: Significant variability in mid-range temperatures suggests fluctuating demands.
    - **Daytime Analysis**: Variability in load forecasts throughout the day correlates with daily temperature changes.
    """)
elif selected_iso_id == "PJM":
    st.markdown("""
    ### PJM Analysis
    - **Consistent Pattern**: The strong correlation shows a predictable relationship between temperature and load forecast.
    - **Daily Peaks**: Peak demands during warmer parts of the day are critical for planning.
    """)

# Create the bubble chart
if not filtered_df.empty:
    fig = px.scatter(filtered_df, x="temperature_celsius", y="load_forecast",
                     size="forecast_difference", color="load_date",
                     hover_name="load_date",
                     title=f"Bubble Chart: Today's Temperature vs. Load Forecast for {selected_iso_id}",
                     labels={
                         "temperature_celsius": "Today's Temperature (°C)",
                         "load_forecast": "Load Forecast",
                         "forecast_difference": "Difference from Average Forecast"
                     },
                     size_max=20)
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected ISO ID.")

# ****************************************************

# Box plot 

# Define the regions or ISO IDs you're comparing
iso_ids = df['iso_id'].unique()

# Create a dropdown to select the type of comparison: Temperature or Load Forecast
comparison_type = st.selectbox("Select Type of Comparison", ["Temperature", "Load Forecast"])

# Based on the selection, create the appropriate boxplot
if comparison_type == "Temperature":
    # Create the boxplot for temperature
    fig = px.box(df, x="iso_id", y="temperature_celsius",
                 title="Comparison of Temperature Across Regions",
                 labels={"iso_id": "Region", "temperature_celsius": "Temperature (°C)"})
    st.plotly_chart(fig)
elif comparison_type == "Load Forecast":
    # Create the boxplot for load forecast
    fig = px.box(df, x="iso_id", y="load_forecast",
                 title="Comparison of Load Forecast Across Regions",
                 labels={"iso_id": "Region", "load_forecast": "Load Forecast"})
    st.plotly_chart(fig)


# ****************************************************

# """
# Correlation matrix (useless)
# """

# # Create a dropdown to select an iso_id with a unique key
# iso_ids = df['iso_id'].unique()
# selected_iso_id = st.selectbox("Select ISO ID", iso_ids, key='iso_selectbox_heatmap')

# # Filter the DataFrame for the selected iso_id
# filtered_df = df[df['iso_id'] == selected_iso_id]

# # Prepare the data: calculate yesterday's load forecast and the forecast difference
# if not filtered_df.empty:
#     filtered_df['yesterday_load_forecast'] = filtered_df['load_forecast'].shift(1)
#     average_forecast = filtered_df['load_forecast'].mean()
#     filtered_df['forecast_difference'] = filtered_df['load_forecast'] - average_forecast

#     # Select relevant features for correlation analysis
#     features = ['temperature_celsius', 'load_forecast', 'yesterday_load_forecast', 'forecast_difference']
#     correlation_data = filtered_df[features]

#     # Calculate the correlation matrix
#     correlation_matrix = correlation_data.corr()

#     # Create a heatmap using Plotly
#     fig = px.imshow(correlation_matrix,
#                      text_auto=True,
#                      color_continuous_scale='Viridis',
#                      title=f'Correlation Matrix Heatmap for {selected_iso_id}',
#                      labels=dict(x="Features", y="Features", color="Correlation Coefficient"))

#     # Display the heatmap in Streamlit
#     st.plotly_chart(fig)

#     # Alternatively, create a heatmap using Seaborn and Matplotlib
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", square=True, cbar_kws={"shrink": .8})
#     plt.title(f'Correlation Matrix Heatmap for {selected_iso_id} (Seaborn)')
#     st.pyplot(plt)
# else:
#     st.write("No data available for the selected ISO ID.")

# ****************************************************
# """
# 1. Ce que tu en comprends:  
# 2. Les différents choix que tu as fait pour en arriver là, pourquoi?
# 3. Ce que tu aurais pu faire différemment, etc.
# """

st.header("Statistical analysis between regions")
# Select regions for comparison
st.subheader("T-Test of temperature between two regions")
regions = df['iso_id'].unique()
selected_regions = st.multiselect("Choose regions to compare", regions)

# Calculate and display t-tests
if len(selected_regions) == 2:
    temp_data = [df[df['iso_id'] == region]['temperature_celsius'] for region in selected_regions]

    # Conduct t-test
    t_stat, p_value = ttest_ind(temp_data[0], temp_data[1], equal_var=False)

    # Display results
    st.markdown(f"T-Test Results between {selected_regions[0]} and {selected_regions[1]}")
    st.markdown(f"- **T-Statistic**: {t_stat:.4f}")
    st.markdown(f"- **P-Value**: {p_value:.4f}")

    # Interpretation
    if p_value < 0.05:
        st.markdown(f"Conclusion: There is a significant difference in temperatures between {selected_regions[0]} and {selected_regions[1]}.")
    else:
        st.markdown(f"Conclusion: There is no significant difference in temperatures between {selected_regions[0]} and {selected_regions[1]}. This means that...")

elif len(selected_regions) > 2:
    st.warning("Please select only two regions for comparison.")
else:
    st.info("Select two regions to perform the t-test.")

st.header("Conclusion & reflection")

st.markdown(
    """
    If we had a longer time period, we could have investigated the seasonal trends in temperature and load forecast for each region.
    We could have identified the peak load periods and the corresponding temperature ranges for each region. And the meteorological data could have been further analyzed to understand the impact of weather conditions on energy consumption.
    event that caused it. 

    Other interesting data source that could have been use to identify relationship:
    1. 

    Reflection on the way the temperature were computed: 
        - How was the temperature data was computed? Given the surface area of the region, the temperature could have been computed relative to the amount of electricity 
        consume per city. Thus, not only averaging,but also weighting the temperature data could have been more accurate.
    
    Explain why is there more variability in the NYISO region:
    """
)