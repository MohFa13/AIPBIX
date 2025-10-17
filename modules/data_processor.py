import pandas as pd
import numpy as np
import io
from typing import Dict, Any, List, Union
import streamlit as st

class DataProcessor:
    """Handles data loading, processing, and analysis"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def load_dataset(self, uploaded_file) -> pd.DataFrame:
        """Load dataset from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                try:
                    dataset = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    dataset = pd.read_csv(uploaded_file, encoding='latin-1')
            
            elif file_extension in ['xlsx', 'xls']:
                dataset = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data cleaning
            dataset = self._clean_dataset(dataset)
            
            return dataset
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Ensure column names are not empty
        df.columns = [f"Column_{i}" if str(col).strip() == '' else str(col) for i, col in enumerate(df.columns)]
        
        return df
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed information about columns"""
        info_data = []
        
        for col in df.columns:
            col_info = {
                'Column': col,
                'Data_Type': str(df[col].dtype),
                'Non_Null_Count': df[col].notna().sum(),
                'Null_Count': df[col].isna().sum(),
                'Null_Percentage': f"{(df[col].isna().sum() / len(df)) * 100:.2f}%",
                'Unique_Values': df[col].nunique(),
                'Sample_Values': ', '.join(str(x) for x in df[col].dropna().head(3).values)
            }
            
            # Add specific stats based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': round(df[col].mean(), 2) if pd.notna(df[col].mean()) else None
                })
            
            info_data.append(col_info)
        
        return pd.DataFrame(info_data)
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and categorize data types for Power BI"""
        type_mapping = {}
        
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                type_mapping[col] = 'integer'
            elif pd.api.types.is_float_dtype(df[col]):
                type_mapping[col] = 'decimal'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_mapping[col] = 'dateTime'
            elif pd.api.types.is_bool_dtype(df[col]):
                type_mapping[col] = 'boolean'
            else:
                # Check if string column could be a date
                if self._could_be_date(df[col]):
                    type_mapping[col] = 'dateTime'
                else:
                    type_mapping[col] = 'string'
        
        return type_mapping
    
    def _could_be_date(self, series: pd.Series) -> bool:
        """Check if a string series could be a date column"""
        sample_values = series.dropna().head(10)
        
        if len(sample_values) == 0:
            return False
        
        date_indicators = ['date', 'time', 'created', 'updated', 'year', 'month', 'day']
        series_name = str(series.name).lower()
        
        # Check column name for date indicators
        if any(indicator in series_name for indicator in date_indicators):
            return True
        
        # Try to parse a few values as dates
        try:
            pd.to_datetime(sample_values.iloc[0])
            return True
        except:
            return False
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'text_columns': len(df.select_dtypes(include=[object]).columns),
            'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_data_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum()
        }
        
        return summary
    
    def identify_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential relationships between columns"""
        relationships = []
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Check for potential foreign key relationships
                if self._could_be_related(df[col1], df[col2]):
                    relationship = {
                        'from_column': col1,
                        'to_column': col2,
                        'relationship_type': 'many_to_one',
                        'confidence': self._calculate_relationship_confidence(df[col1], df[col2])
                    }
                    relationships.append(relationship)
        
        return sorted(relationships, key=lambda x: x['confidence'], reverse=True)
    
    def _could_be_related(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if two columns could have a relationship"""
        # Simple heuristic: check if one column's values are mostly contained in the other
        unique1 = set(series1.dropna().unique())
        unique2 = set(series2.dropna().unique())
        
        if len(unique1) == 0 or len(unique2) == 0:
            return False
        
        # Check containment in both directions
        containment1 = len(unique1.intersection(unique2)) / len(unique1)
        containment2 = len(unique2.intersection(unique1)) / len(unique2)
        
        return containment1 > 0.3 or containment2 > 0.3
    
    def _calculate_relationship_confidence(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate confidence score for relationship"""
        unique1 = set(series1.dropna().unique())
        unique2 = set(series2.dropna().unique())
        
        if len(unique1) == 0 or len(unique2) == 0:
            return 0.0
        
        intersection = unique1.intersection(unique2)
        union = unique1.union(unique2)
        
        # Jaccard similarity
        return len(intersection) / len(union)
