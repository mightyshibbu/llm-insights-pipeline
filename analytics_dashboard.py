import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
import plotly.express as px
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Email Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize ChromaDB client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

def load_email_data():
    """Load email data from ChromaDB"""
    try:
        client = get_chroma_client()
        collection = client.get_collection("email_summaries")
        
        # Get all items
        results = collection.get(include=["metadatas", "documents"])
        
        if not results['ids']:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame({
            'id': results['ids'],
            **{f'metadata_{k}': [d.get(k) for d in results['metadatas']] 
               for k in results['metadatas'][0].keys()},
            'content': results['documents']
        })
        
        # Convert timestamp to datetime if exists
        if 'metadata_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['metadata_timestamp'], unit='s')
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def display_metrics(df):
    """Display key metrics"""
    if df.empty:
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(df))
    
    with col2:
        if 'metadata_sender' in df.columns:
            st.metric("Unique Senders", df['metadata_sender'].nunique())
    
    with col3:
        if 'metadata_incident_type' in df.columns:
            st.metric("Incident Types", df['metadata_incident_type'].nunique())
    
    with col4:
        if 'metadata_severity' in df.columns:
            st.metric("High Severity", 
                     df[df['metadata_severity'] == 'High'].shape[0])

def plot_timeline(df):
    """Plot email timeline"""
    if df.empty or 'timestamp' not in df.columns:
        return
        
    # Group by date
    timeline = df.set_index('timestamp').resample('D').size()
    fig = px.line(timeline, title='Emails Over Time', 
                  labels={'value': 'Number of Emails', 'timestamp': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

def plot_incident_types(df):
    """Plot incident type distribution"""
    if df.empty or 'metadata_incident_type' not in df.columns:
        return
        
    incident_counts = df['metadata_incident_type'].value_counts()
    fig = px.pie(incident_counts, names=incident_counts.index, 
                 values=incident_counts.values, title='Incident Type Distribution')
    st.plotly_chart(fig, use_container_width=True)

def plot_severity_distribution(df):
    """Plot severity distribution"""
    if df.empty or 'metadata_severity' not in df.columns:
        return
        
    severity_counts = df['metadata_severity'].value_counts()
    fig = px.bar(severity_counts, x=severity_counts.index, y=severity_counts.values,
                 title='Severity Distribution', 
                 labels={'y': 'Count', 'index': 'Severity'})
    st.plotly_chart(fig, use_container_width=True)

def display_data_table(df):
    """Display raw data in a table"""
    if df.empty:
        return
        
    # Select columns to display
    cols_to_show = ['id', 'timestamp', 'metadata_sender', 'metadata_incident_type', 
                   'metadata_severity', 'content']
    cols_to_show = [col for col in cols_to_show if col in df.columns]
    
    st.subheader('Email Data')
    st.dataframe(df[cols_to_show], use_container_width=True)

def main():
    st.title('📊 Email Analytics Dashboard')
    
    # Load data
    df = load_email_data()
    
    if df.empty:
        st.warning("No email data found in the database.")
        return
    
    # Display metrics
    display_metrics(df)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Incident Analysis", "Severity Analysis", "Raw Data"])
    
    with tab1:
        st.header("Email Overview")
        plot_timeline(df)
    
    with tab2:
        st.header("Incident Analysis")
        plot_incident_types(df)
    
    with tab3:
        st.header("Severity Analysis")
        plot_severity_distribution(df)
    
    with tab4:
        st.header("Email Data")
        display_data_table(df)

if __name__ == "__main__":
    main()
