import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import json
from pathlib import Path
import traceback

from modules.data_processor import DataProcessor
from modules.ai_models import AIModels
from modules.pbit_generator import PBITGenerator
from modules.visualization_analyzer import VisualizationAnalyzer
from modules.tiny_recursive_agent import TinyRecursiveAgent
from utils.file_utils import FileUtils

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'data_analysis' not in st.session_state:
    st.session_state.data_analysis = None
if 'ai_models_loaded' not in st.session_state:
    st.session_state.ai_models_loaded = False

def main():
    st.set_page_config(
        page_title="AI Power BI Dashboard Generator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Power BI Dashboard Generator")
    st.markdown("Generate Power BI templates from text prompts and datasets using AI agents")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a step:",
            ["📁 Upload Dataset", "🤖 AI Analysis", "📝 Design Prompt", "🎨 Generate Dashboard"]
        )
    
    # Initialize components
    try:
        if not st.session_state.ai_models_loaded:
            with st.spinner("Loading AI models..."):
                st.session_state.ai_models = AIModels()
                st.session_state.tiny_agent = TinyRecursiveAgent()
                st.session_state.ai_models_loaded = True
            st.success("AI models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading AI models: {str(e)}")
        st.error("Please check your internet connection and try again.")
        return
    
    if page == "📁 Upload Dataset":
        upload_dataset_page()
    elif page == "🤖 AI Analysis":
        ai_analysis_page()
    elif page == "📝 Design Prompt":
        design_prompt_page()
    elif page == "🎨 Generate Dashboard":
        generate_dashboard_page()

def upload_dataset_page():
    st.header("📁 Dataset Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            data_processor = DataProcessor()
            
            with st.spinner("Processing dataset..."):
                dataset = data_processor.load_dataset(uploaded_file)
                st.session_state.dataset = dataset
                st.session_state.data_uploaded = True
            
            st.success(f"Dataset uploaded successfully! Shape: {dataset.shape}")
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(dataset.head(10))
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", dataset.shape[0])
            with col2:
                st.metric("Columns", dataset.shape[1])
            with col3:
                st.metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Column information
            st.subheader("Column Information")
            col_info = data_processor.get_column_info(dataset)
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            st.error(traceback.format_exc())

def ai_analysis_page():
    st.header("🤖 AI-Powered Data Analysis")
    
    if not st.session_state.data_uploaded:
        st.warning("Please upload a dataset first.")
        return
    
    if st.button("Run AI Analysis"):
        try:
            with st.spinner("Running AI analysis on your dataset..."):
                analyzer = VisualizationAnalyzer(st.session_state.ai_models)
                
                # Perform comprehensive data analysis
                analysis_results = analyzer.analyze_dataset(st.session_state.dataset)
                st.session_state.data_analysis = analysis_results
                
            st.success("AI analysis completed!")
            
            # Display analysis results
            st.subheader("📊 Data Analysis Results")
            
            # Data summary
            st.write("**Dataset Summary:**")
            st.write(analysis_results.get('summary', 'No summary available'))
            
            # Data types and patterns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Detected Data Types:**")
                for col, dtype in analysis_results.get('data_types', {}).items():
                    st.write(f"- {col}: {dtype}")
            
            with col2:
                st.write("**Key Statistics:**")
                for stat, value in analysis_results.get('key_stats', {}).items():
                    st.write(f"- {stat}: {value}")
            
            # Visualization recommendations
            st.subheader("🎨 Recommended Visualizations")
            recommendations = analysis_results.get('viz_recommendations', [])
            
            for i, rec in enumerate(recommendations):
                with st.expander(f"Recommendation {i+1}: {rec.get('title', 'Visualization')}"):
                    st.write(f"**Type:** {rec.get('type', 'Unknown')}")
                    st.write(f"**Columns:** {', '.join(rec.get('columns', []))}")
                    st.write(f"**Reasoning:** {rec.get('reasoning', 'No reasoning provided')}")
                    st.write(f"**Priority:** {rec.get('priority', 'Medium')}")
            
        except Exception as e:
            st.error(f"Error during AI analysis: {str(e)}")
            st.error(traceback.format_exc())
    
    # Display existing analysis if available
    if st.session_state.data_analysis:
        st.subheader("Previous Analysis Results")
        st.json(st.session_state.data_analysis)

def design_prompt_page():
    st.header("📝 Design Your Dashboard")
    
    if not st.session_state.data_uploaded:
        st.warning("Please upload a dataset first.")
        return
    
    st.write("Describe your desired dashboard design and visualizations:")
    
    # Design prompt input
    design_prompt = st.text_area(
        "Dashboard Design Prompt",
        placeholder="Example: Create a sales dashboard with monthly trends, top products by revenue, and regional performance comparison. Use blue color scheme and include KPI cards for total sales, growth rate, and profit margin.",
        height=150
    )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Blue", "Green", "Red", "Purple", "Orange", "Custom"]
        )
        
        dashboard_type = st.selectbox(
            "Dashboard Type",
            ["Executive Summary", "Operational", "Analytical", "Custom"]
        )
    
    with col2:
        include_kpis = st.checkbox("Include KPI Cards", value=True)
        include_filters = st.checkbox("Include Interactive Filters", value=True)
    
    # Store design preferences in session state
    st.session_state.design_prompt = design_prompt
    st.session_state.color_scheme = color_scheme
    st.session_state.dashboard_type = dashboard_type
    st.session_state.include_kpis = include_kpis
    st.session_state.include_filters = include_filters
    
    if design_prompt and st.button("Process Design Requirements"):
        try:
            with st.spinner("Processing design requirements with AI..."):
                # Use TinyRecursive agent for multi-step reasoning
                agent_response = st.session_state.tiny_agent.process_design_request(
                    design_prompt,
                    st.session_state.dataset,
                    st.session_state.data_analysis
                )
                
                st.session_state.design_analysis = agent_response
            
            st.success("Design requirements processed!")
            
            # Display agent analysis
            st.subheader("🧠 AI Agent Analysis")
            st.write("**Design Interpretation:**")
            st.write(agent_response.get('interpretation', 'No interpretation available'))
            
            st.write("**Recommended Layout:**")
            layout = agent_response.get('layout', {})
            for section, details in layout.items():
                st.write(f"- **{section}:** {details}")
            
        except Exception as e:
            st.error(f"Error processing design requirements: {str(e)}")

def generate_dashboard_page():
    st.header("🎨 Generate Power BI Dashboard")
    
    if not st.session_state.data_uploaded:
        st.warning("Please upload a dataset first.")
        return
    
    if not hasattr(st.session_state, 'design_prompt') or not st.session_state.design_prompt:
        st.warning("Please provide a design prompt first.")
        return
    
    if st.button("Generate PBIT Template"):
        try:
            with st.spinner("Generating Power BI template..."):
                # Initialize PBIT generator
                pbit_generator = PBITGenerator()
                
                # Generate the PBIT file
                pbit_content = pbit_generator.generate_pbit(
                    dataset=st.session_state.dataset,
                    data_analysis=st.session_state.data_analysis,
                    design_prompt=st.session_state.design_prompt,
                    design_analysis=getattr(st.session_state, 'design_analysis', {}),
                    color_scheme=getattr(st.session_state, 'color_scheme', 'Blue'),
                    include_kpis=getattr(st.session_state, 'include_kpis', True),
                    include_filters=getattr(st.session_state, 'include_filters', True)
                )
                
                st.session_state.pbit_content = pbit_content
            
            st.success("PBIT template generated successfully!")
            
            # Provide download button
            st.download_button(
                label="📥 Download PBIT Template",
                data=st.session_state.pbit_content,
                file_name="ai_generated_dashboard.pbit",
                mime="application/octet-stream"
            )
            
            # Generate PBIX with data
            if st.button("Generate PBIX with Data"):
                try:
                    with st.spinner("Creating PBIX file with your data..."):
                        pbix_content = pbit_generator.create_pbix_from_pbit(
                            st.session_state.pbit_content,
                            st.session_state.dataset
                        )
                        
                        st.session_state.pbix_content = pbix_content
                    
                    st.success("PBIX file generated successfully!")
                    
                    st.download_button(
                        label="📥 Download PBIX File",
                        data=st.session_state.pbix_content,
                        file_name="ai_generated_dashboard.pbix",
                        mime="application/octet-stream"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating PBIX: {str(e)}")
            
            # Display generation summary
            st.subheader("📋 Generation Summary")
            st.write("**Generated Components:**")
            st.write("- Data model schema with tables and relationships")
            st.write("- Visualization layouts based on AI recommendations")
            st.write("- Power Query transformations")
            st.write("- Custom color themes and formatting")
            
        except Exception as e:
            st.error(f"Error generating dashboard: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
