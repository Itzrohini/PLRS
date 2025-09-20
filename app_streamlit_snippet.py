
# streamlit app: simple demo
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Use st.cache_resource for a trained model
@st.cache_resource
def load_model():
    return joblib.load('models/pipeline_model.joblib')

# Use st.cache_data for static data like CSV
@st.cache_data
def load_data():
    return pd.read_csv('oulad_merged.csv')

# Use st.cache_data for feature list
@st.cache_data
def load_features():
    return joblib.load('models/used_features.joblib')

# Load all necessary artifacts
pipe = load_model()
df = load_data()
features = load_features()

def recommend_for_student(student_id, df, k=5):
    interactions = df.groupby(['id_student','code_module'])['total_clicks'].sum().reset_index()
    pop = interactions.groupby('code_module')['total_clicks'].sum().sort_values(ascending=False)
    return pop.head(k).index.tolist()

st.title('Personalized Learning Demo')
student_id = st.number_input('Enter student id', min_value=int(df['id_student'].min()), max_value=int(df['id_student'].max()), value=11391)

row = df[df['id_student']==student_id]
if row.empty:
    st.write('Student not found')
else:
    # Feature engineering to match pipeline expectations
    row['log_total_clicks'] = np.log1p(row['total_clicks'])
    row['sqrt_days_active'] = np.sqrt(row['days_active'].clip(lower=0))
    
    # Aggregates must be calculated correctly
    agg_cols_demo = ['total_clicks', 'days_active', 'num_of_prev_attempts', 'studied_credits']
    student_agg_row = row.groupby('id_student').agg(
        total_clicks_sum=('total_clicks', 'sum'),
        total_clicks_mean=('total_clicks', 'mean'),
        days_active_sum=('days_active', 'sum'),
        days_active_mean=('days_active', 'mean'),
        num_of_prev_attempts_sum=('num_of_prev_attempts', 'sum'),
        num_of_prev_attempts_mean=('num_of_prev_attempts', 'mean'),
        studied_credits_sum=('studied_credits', 'sum'),
        studied_credits_mean=('studied_credits', 'mean')
    ).reset_index()
    
    # Merge aggregates into the student's row
    row = row.merge(student_agg_row, on='id_student', how='left')
    
    # Select final features
    Xs = row[features].head(1)
    
    pred = pipe.predict_proba(Xs)[0,1]
    st.metric('Predicted pass probability', f"{pred:.3f}")
    
    recs = recommend_for_student(student_id, df)
    st.write('Recommended modules:')
    st.write(recs)
