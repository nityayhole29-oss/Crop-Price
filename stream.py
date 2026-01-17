import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# --- CONFIGURATION AND PAGE SETUP ---
st.set_page_config(
    page_title="AgriPredict: Crop Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for an eye-catching, professional, and dark-green/gold aesthetic
st.markdown("""
<style>
    /* Main body background and font */
    body {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f0f7f4; /* Light clean green/gray background */
    }

    /* Header and Title Styling */
    .css-1dp5r8l {
        font-weight: 900;
        color: #1b4d3e; /* Deep Forest Green */
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    /* KPI Metrics Styling (Gold/Green) */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #d4af37; /* Gold for value */
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #1b4d3e; /* Deep Forest Green for label */
        font-weight: 600;
    }

    /* Prediction button style */
    .stButton>button {
        background-color: #38761d; /* Darker leaf green */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: 700;
        transition: all 0.3s;
        width: 100%;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #1b4d3e;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Prediction Result Box */
    .prediction-result {
        border-radius: 12px;
        background-color: #e6f7ee;
        border: 2px solid #38761d;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-price {
        font-size: 5rem;
        color: #38761d;
        font-weight: 900;
        line-height: 1.1;
    }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL VARIABLES ---
DATA_FILE = 'data_season.csv'

# --- 2. DATA LOADING AND MODEL TRAINING (Cached for Efficiency) ---

@st.cache_resource
def load_data_and_train_model(data_path):
    """Loads data, trains the ML pipeline, and returns the model, R2 score, and DataFrame."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: The data file '{data_path}' was not found. Please ensure it is in the same directory.")
        return None, 0, None

    # Standardize column names based on file inspection
    df.columns = [col.strip().replace('yeilds', 'Yield').replace('Crops', 'Crop') for col in df.columns]

    REQUIRED_COLUMNS = ['Area', 'Temperature', 'Yield', 'Crop', 'price', 'Year', 'Season']
    
    # Check for missing columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Data Error: Required columns are missing or misnamed: {missing_cols}")
        return None, 0, None

    # FIX: Explicitly convert 'price' to numeric, coercing errors to NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Drop any row that now has NaN in a required column
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    
    # Check if we have any data left to train on
    if df.empty:
        st.error("Data Error: After cleaning, the dataset is empty. Check your CSV formatting.")
        return None, 0, None

    df['Year'] = df['Year'].astype(int)
    
    # ML Pipeline Setup
    X = df[['Area', 'Temperature', 'Yield', 'Crop']]
    y = df['price']

    categorical_features = ['Crop']
    
    # Define the preprocessor to handle categorical features (OneHotEncoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full ML pipeline with Random Forest Regressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred) * 100

    return model_pipeline, r2, df

# Execute loading and training
ml_pipeline, r2_score_val, data_df = load_data_and_train_model(DATA_FILE)

# --- SITE HEADER ---

st.title("🌱 AgriPredict: Market Intelligence Dashboard")
st.markdown("A data-driven platform for agricultural insights and price forecasting using Machine Learning.")

# Check for model/data failure after execution
if ml_pipeline is None or data_df is None or data_df.empty:
    st.stop() # Stops execution if data loading or cleaning failed


# --- TABS FOR DASHBOARD AND PREDICTION ---
tab_dashboard, tab_predict = st.tabs(["📊 Market Dashboard", "🔮 Price Prediction Tool"])

# --- TAB 1: MARKET DASHBOARD ---
with tab_dashboard:
    
    st.header("Global Market Trends")
    st.markdown("Explore key trends, feature distribution, and the historical performance of crop prices.")
    
    # 1. Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    
    avg_price = data_df['price'].mean()
    avg_yield = data_df['Yield'].mean()
    unique_crops = data_df['Crop'].nunique()
    
    col1.metric("Average Price (₹/Unit)", f"₹{avg_price:,.0f}")
    col2.metric("Average Yield (Tons)", f"{avg_yield:,.0f} T")
    col3.metric("Total Crops Tracked", unique_crops)
    col4.metric("Model Reliability (R²)", f"{r2_score_val:.1f}%")

    st.markdown("---")
    
    # 2. Main Visualization Row (Line Chart and Pie Chart)
    col_chart_1, col_chart_2 = st.columns([2, 1])

    with col_chart_1:
        st.subheader("Historical Price Trend Over Time")
        # Group by year and calculate the average price
        price_trend = data_df.groupby('Year')['price'].mean().reset_index()
        fig_line = px.line(
            price_trend, 
            x='Year', 
            y='price', 
            title='Average Crop Price Evolution',
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#38761d']
        )
        fig_line.update_layout(xaxis_title="Year", yaxis_title="Average Price (₹)")
        # Removed use_container_width=True to avoid warning
        st.plotly_chart(fig_line) 
    
    with col_chart_2:
        st.subheader("Crop Distribution by Volume")
        # Pie Chart showing the volume of each crop in the dataset
        crop_counts = data_df['Crop'].value_counts().head(5).reset_index()
        crop_counts.columns = ['Crop', 'Count']
        fig_pie = px.pie(
            crop_counts, 
            values='Count', 
            names='Crop', 
            title='Top 5 Most Tracked Crops',
            color_discrete_sequence=px.colors.sequential.Greens_r # Green color palette
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        # Removed use_container_width=True to avoid warning
        st.plotly_chart(fig_pie) 

    st.markdown("---")

    # 3. Secondary Visualization Row (Scatter Plot and Season Trend)
    col_chart_3, col_chart_4 = st.columns(2)

    with col_chart_3:
        st.subheader("Yield vs. Area Analysis")
        # Scatter plot to show relationship between yield and area, colored by crop type
        fig_scatter = px.scatter(
            data_df, 
            x='Area', 
            y='Yield', 
            color='Crop', 
            hover_data=['price'],
            title='Yield vs. Area by Crop Type',
            opacity=0.6,
            log_x=True, # Log scale helps handle large area differences
            log_y=True,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_scatter.update_layout(xaxis_title="Area (Hectares, Log Scale)", yaxis_title="Yield (Tons, Log Scale)")
        # Removed use_container_width=True to avoid warning
        st.plotly_chart(fig_scatter)
    
    with col_chart_4:
        st.subheader("Seasonal Price Impact")
        # Box plot showing how price varies across seasons
        fig_box = px.box(
            data_df, 
            x='Season', 
            y='price', 
            color='Season',
            title='Price Distribution Across Seasons',
            color_discrete_sequence=px.colors.sequential.Sunsetdark # Using a complementary palette
        )
        fig_box.update_layout(xaxis_title="Season", yaxis_title="Price (₹)")
        # Removed use_container_width=True to avoid warning
        st.plotly_chart(fig_box)


# --- TAB 2: PRICE PREDICTION TOOL ---
with tab_predict:
    
    col_img, col_form = st.columns([1, 2])
    
    with col_img:
        st.image("https://placehold.co/400x350/1b4d3e/d4af37?text=Predictive+Farming", caption="Data-Driven Market Forecasting", use_column_width=True)
        st.markdown("---")
        st.markdown("**Model Details:** Trained on historical data including **Area, Temperature, Yield, and Crop type**. The model uses a robust **Random Forest Regressor** to predict non-linear price fluctuations.")
    
    with col_form:
        st.header("Forecasting Tool")
        st.markdown("Input the four key parameters below to generate a forward-looking price forecast.")
        
        # 1. Categorical Feature: Crop Type
        unique_crops = sorted(data_df['Crop'].unique().tolist())
        selected_crop = st.selectbox(
            "1. Crop Name",
            options=unique_crops,
            index=0,
            help="Select the crop to predict."
        )

        # 2-4. Numerical Features
        col_area, col_temp, col_yield = st.columns(3)
        
        # Calculate robust default/range values based on the cleaned data
        area_median = data_df['Area'].median()
        temp_median = data_df['Temperature'].median()
        yield_median = data_df['Yield'].median()

        with col_area:
            selected_area = st.number_input(
                "2. Area (Hectares)",
                min_value=0.0,
                max_value=data_df['Area'].max() * 1.5,
                value=float(area_median),
                step=100.0,
                format="%.2f",
            )

        with col_temp:
            selected_temp = st.number_input(
                "3. Avg. Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=float(temp_median),
                step=0.1,
                format="%.1f",
            )

        with col_yield:
            selected_yield = st.number_input(
                "4. Expected Yield (Tons)",
                min_value=0.0,
                max_value=data_df['Yield'].max() * 1.5,
                value=float(yield_median),
                step=100.0,
                format="%.2f",
            )

        # Prediction Button
        if st.button("Predict Market Price", key="predict_button_final"):
            
            # --- PREDICTION EXECUTION ---
            input_data = pd.DataFrame([{
                'Area': selected_area,
                'Temperature': selected_temp,
                'Yield': selected_yield,
                'Crop': selected_crop
            }])

            try:
                prediction = ml_pipeline.predict(input_data)[0]
                predicted_price = max(0, prediction)
                formatted_price = f"₹ {predicted_price:,.2f}"

                # Display Result
                st.markdown(f"""
                    <div class='prediction-result'>
                        <p style='font-size: 1.5rem; color: #374151;'>Forecasted Price for **{selected_crop}**:</p>
                        <p class='prediction-price'>{formatted_price}</p>
                        <p style='font-size: 1.0rem; color: #38761d;'>Per unit (based on model training data)</p>
                    </div>  
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: Could not generate forecast. Details: {e}")

# --- SITE FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("AgriPredict ML Dashboard | Developed with Streamlit")
st.sidebar.metric("Dataset Size", f"{len(data_df):,} records")