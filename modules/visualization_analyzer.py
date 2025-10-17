import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re

class VisualizationAnalyzer:
    """Analyzes datasets and recommends optimal visualizations"""
    
    def __init__(self, ai_models):
        self.ai_models = ai_models
        
    def analyze_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis"""
        
        # Basic statistics
        basic_stats = self._get_basic_statistics(dataset)
        
        # Data type analysis
        data_types = self._analyze_data_types(dataset)
        
        # Column analysis
        column_analysis = self._analyze_columns(dataset)
        
        # Pattern detection
        patterns = self._detect_patterns(dataset)
        
        # Visualization recommendations
        viz_recommendations = self._recommend_visualizations(dataset, column_analysis, patterns)
        
        # Generate sample data for LLM analysis
        sample_data = self._prepare_sample_data(dataset)
        
        # Get AI insights
        ai_insights = self.ai_models.analyze_data_with_llm(basic_stats, sample_data)
        
        return {
            'summary': ai_insights.get('summary', 'No summary available'),
            'basic_stats': basic_stats,
            'data_types': data_types,
            'column_analysis': column_analysis,
            'patterns': patterns,
            'viz_recommendations': viz_recommendations,
            'ai_insights': ai_insights.get('insights', []),
            'quality_issues': ai_insights.get('quality_issues', []),
            'key_stats': self._extract_key_statistics(dataset)
        }
    
    def _get_basic_statistics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        return {
            'row_count': len(dataset),
            'column_count': len(dataset.columns),
            'memory_usage_mb': dataset.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(dataset.select_dtypes(include=[np.number]).columns),
            'text_columns': len(dataset.select_dtypes(include=[object]).columns),
            'date_columns': len(dataset.select_dtypes(include=['datetime64']).columns),
            'missing_data_percentage': (dataset.isna().sum().sum() / (len(dataset) * len(dataset.columns))) * 100,
            'duplicate_rows': dataset.duplicated().sum(),
            'unique_values_per_column': dataset.nunique().to_dict()
        }
    
    def _analyze_data_types(self, dataset: pd.DataFrame) -> Dict[str, str]:
        """Analyze and categorize data types"""
        type_analysis = {}
        
        for column in dataset.columns:
            dtype = dataset[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    type_analysis[column] = 'Integer'
                else:
                    type_analysis[column] = 'Decimal'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_analysis[column] = 'DateTime'
            elif pd.api.types.is_bool_dtype(dtype):
                type_analysis[column] = 'Boolean'
            else:
                # Further analyze text columns
                if self._is_categorical(dataset[column]):
                    type_analysis[column] = 'Categorical'
                elif self._could_be_date(dataset[column]):
                    type_analysis[column] = 'Date (Text)'
                elif self._is_id_column(dataset[column]):
                    type_analysis[column] = 'ID/Key'
                else:
                    type_analysis[column] = 'Text'
        
        return type_analysis
    
    def _analyze_columns(self, dataset: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze individual columns in detail"""
        column_analysis = {}
        
        for column in dataset.columns:
            col_data = dataset[column]
            
            analysis = {
                'data_type': str(col_data.dtype),
                'null_count': col_data.isna().sum(),
                'null_percentage': (col_data.isna().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100,
                'sample_values': col_data.dropna().head(5).tolist()
            }
            
            # Add specific analysis based on data type
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'distribution_type': self._analyze_distribution(col_data)
                })
            elif pd.api.types.is_object_dtype(col_data):
                value_counts = col_data.value_counts().head(10)
                analysis.update({
                    'top_values': value_counts.to_dict(),
                    'avg_length': col_data.astype(str).str.len().mean(),
                    'is_categorical': self._is_categorical(col_data)
                })
            
            column_analysis[column] = analysis
        
        return column_analysis
    
    def _detect_patterns(self, dataset: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect common patterns in the dataset"""
        patterns = {
            'time_series': [],
            'hierarchical': [],
            'geographical': [],
            'categorical_relationships': [],
            'numeric_correlations': []
        }
        
        # Detect time series columns
        for col in dataset.columns:
            if pd.api.types.is_datetime64_any_dtype(dataset[col]) or self._could_be_date(dataset[col]):
                patterns['time_series'].append(col)
        
        # Detect hierarchical relationships (e.g., Category -> Subcategory)
        text_columns = dataset.select_dtypes(include=[object]).columns
        for i, col1 in enumerate(text_columns):
            for col2 in text_columns[i+1:]:
                if self._is_hierarchical_relationship(dataset[col1], dataset[col2]):
                    patterns['hierarchical'].append(f"{col1} -> {col2}")
        
        # Detect geographical columns
        for col in dataset.columns:
            if self._is_geographical_column(col, dataset[col]):
                patterns['geographical'].append(col)
        
        # Detect numeric correlations
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = dataset[numeric_cols].corr()
            high_corr = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        high_corr.append(f"{col1} <-> {col2} ({corr:.2f})")
            patterns['numeric_correlations'] = high_corr
        
        return patterns
    
    def _recommend_visualizations(self, dataset: pd.DataFrame, column_analysis: Dict, patterns: Dict) -> List[Dict[str, Any]]:
        """Recommend optimal visualizations based on data analysis"""
        recommendations = []
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = dataset.select_dtypes(include=[object]).columns.tolist()
        date_cols = patterns['time_series']
        
        # Time series visualizations
        if date_cols and numeric_cols:
            for date_col in date_cols[:2]:  # Limit to 2 date columns
                for num_col in numeric_cols[:2]:  # Limit to 2 numeric columns
                    recommendations.append({
                        'type': 'Line Chart',
                        'title': f'{num_col} over Time',
                        'columns': [date_col, num_col],
                        'reasoning': f'Time series data showing trend of {num_col} over {date_col}',
                        'priority': 'High',
                        'chart_type': 'lineChart'
                    })
        
        # Categorical comparisons
        suitable_categorical = [col for col in text_cols if dataset[col].nunique() <= 20]
        if suitable_categorical and numeric_cols:
            for cat_col in suitable_categorical[:2]:
                for num_col in numeric_cols[:2]:
                    recommendations.append({
                        'type': 'Bar Chart',
                        'title': f'{num_col} by {cat_col}',
                        'columns': [cat_col, num_col],
                        'reasoning': f'Compare {num_col} across different {cat_col} categories',
                        'priority': 'High',
                        'chart_type': 'columnChart'
                    })
        
        # Distribution visualizations
        for num_col in numeric_cols[:3]:
            recommendations.append({
                'type': 'Histogram',
                'title': f'Distribution of {num_col}',
                'columns': [num_col],
                'reasoning': f'Show the distribution pattern of {num_col}',
                'priority': 'Medium',
                'chart_type': 'histogram'
            })
        
        # Pie charts for categorical data with reasonable categories
        pie_suitable = [col for col in suitable_categorical if dataset[col].nunique() <= 8]
        if pie_suitable and numeric_cols:
            recommendations.append({
                'type': 'Pie Chart',
                'title': f'{numeric_cols[0]} by {pie_suitable[0]}',
                'columns': [pie_suitable[0], numeric_cols[0]],
                'reasoning': f'Show proportion of {numeric_cols[0]} across {pie_suitable[0]} categories',
                'priority': 'Medium',
                'chart_type': 'pieChart'
            })
        
        # Correlation matrix for numeric data
        if len(numeric_cols) >= 3:
            recommendations.append({
                'type': 'Correlation Matrix',
                'title': 'Numeric Correlations',
                'columns': numeric_cols[:5],  # Limit to 5 columns
                'reasoning': 'Show relationships between numeric variables',
                'priority': 'Medium',
                'chart_type': 'matrix'
            })
        
        # KPI Cards for key metrics
        key_metrics = self._identify_key_metrics(dataset, numeric_cols)
        for metric in key_metrics[:4]:  # Limit to 4 KPIs
            recommendations.append({
                'type': 'KPI Card',
                'title': f'Total {metric}',
                'columns': [metric],
                'reasoning': f'Key performance indicator for {metric}',
                'priority': 'High',
                'chart_type': 'card'
            })
        
        # Data table for detailed view
        recommendations.append({
            'type': 'Data Table',
            'title': 'Detailed Data View',
            'columns': dataset.columns.tolist()[:10],  # Limit to 10 columns
            'reasoning': 'Provide detailed tabular view of the data',
            'priority': 'Low',
            'chart_type': 'table'
        })
        
        return recommendations
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Determine if a text column is categorical"""
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.5 and series.nunique() <= 50
    
    def _could_be_date(self, series: pd.Series) -> bool:
        """Check if a column could be a date"""
        if series.dtype != 'object':
            return False
        
        # Check column name
        date_indicators = ['date', 'time', 'created', 'updated', 'year', 'month', 'day']
        if any(indicator in str(series.name).lower() for indicator in date_indicators):
            return True
        
        # Try parsing sample values
        sample = series.dropna().head(5)
        date_count = 0
        for value in sample:
            try:
                pd.to_datetime(str(value))
                date_count += 1
            except:
                pass
        
        return date_count >= len(sample) * 0.6  # 60% or more parseable as dates
    
    def _is_id_column(self, series: pd.Series) -> bool:
        """Check if column is likely an ID or key column"""
        if series.dtype != 'object':
            return False
        
        # Check column name
        id_indicators = ['id', 'key', 'code', 'identifier', 'ref']
        col_name = str(series.name).lower()
        if any(indicator in col_name for indicator in id_indicators):
            return True
        
        # Check uniqueness
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.95:  # Very high uniqueness
            return True
        
        # Check pattern (e.g., alphanumeric codes)
        sample = series.dropna().head(10)
        pattern_count = 0
        for value in sample:
            if re.match(r'^[A-Za-z0-9_-]+$', str(value)) and len(str(value)) > 3:
                pattern_count += 1
        
        return pattern_count >= len(sample) * 0.7
    
    def _analyze_distribution(self, series: pd.Series) -> str:
        """Analyze the distribution type of numeric data"""
        if len(series.dropna()) < 3:
            return 'Insufficient data'
        
        try:
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            if abs(skewness) < 0.5:
                return 'Normal-like'
            elif skewness > 1:
                return 'Right-skewed'
            elif skewness < -1:
                return 'Left-skewed'
            elif kurtosis > 3:
                return 'Heavy-tailed'
            else:
                return 'Moderate distribution'
        except:
            return 'Unknown'
    
    def _is_hierarchical_relationship(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if two categorical columns have a hierarchical relationship"""
        try:
            # Create a mapping of series1 values to series2 values
            mapping = {}
            for v1, v2 in zip(series1.dropna(), series2.dropna()):
                if v1 not in mapping:
                    mapping[v1] = set()
                mapping[v1].add(v2)
            
            # Check if each value in series1 consistently maps to few values in series2
            consistent_mapping = 0
            for v1, v2_set in mapping.items():
                if len(v2_set) <= 3:  # Each category maps to 3 or fewer subcategories
                    consistent_mapping += 1
            
            return consistent_mapping >= len(mapping) * 0.8  # 80% consistency
        except:
            return False
    
    def _is_geographical_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column contains geographical data"""
        geo_indicators = ['country', 'state', 'city', 'region', 'location', 'address', 
                         'zip', 'postal', 'lat', 'lng', 'longitude', 'latitude']
        
        col_name_lower = col_name.lower()
        if any(indicator in col_name_lower for indicator in geo_indicators):
            return True
        
        # Check for common geographical values
        if series.dtype == 'object':
            sample_values = series.dropna().head(10).astype(str).str.lower()
            geo_values = ['usa', 'united states', 'california', 'new york', 'texas', 
                         'canada', 'uk', 'france', 'germany', 'japan', 'china']
            
            geo_count = sum(1 for val in sample_values if any(geo in val for geo in geo_values))
            return geo_count >= len(sample_values) * 0.3
        
        return False
    
    def _identify_key_metrics(self, dataset: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Identify key metrics suitable for KPI cards"""
        key_metrics = []
        
        # Prioritize columns with specific names
        priority_indicators = ['sales', 'revenue', 'profit', 'cost', 'price', 'amount', 
                             'total', 'count', 'quantity', 'volume', 'rate', 'percentage']
        
        for col in numeric_cols:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in priority_indicators):
                key_metrics.append(col)
        
        # Add other numeric columns if we don't have enough
        for col in numeric_cols:
            if col not in key_metrics:
                key_metrics.append(col)
        
        return key_metrics
    
    def _prepare_sample_data(self, dataset: pd.DataFrame) -> str:
        """Prepare sample data for LLM analysis"""
        sample = dataset.head(3)
        return sample.to_string(max_cols=10, max_rows=3)
    
    def _extract_key_statistics(self, dataset: pd.DataFrame) -> Dict[str, str]:
        """Extract key statistics for display"""
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        stats = {}
        
        if len(numeric_cols) > 0:
            # Find column with highest sum (likely a key metric)
            sums = dataset[numeric_cols].sum().sort_values(ascending=False)
            if len(sums) > 0:
                top_metric = sums.index[0]
                stats[f'Highest Total ({top_metric})'] = f"{sums.iloc[0]:,.2f}"
        
        # Data completeness
        completeness = ((dataset.notna().sum().sum()) / (len(dataset) * len(dataset.columns))) * 100
        stats['Data Completeness'] = f"{completeness:.1f}%"
        
        # Most common category
        text_cols = dataset.select_dtypes(include=[object]).columns
        if len(text_cols) > 0:
            for col in text_cols[:2]:  # Check first 2 text columns
                if dataset[col].nunique() <= 20:  # Reasonable for categories
                    most_common = dataset[col].mode().iloc[0] if len(dataset[col].mode()) > 0 else 'N/A'
                    stats[f'Most Common {col}'] = str(most_common)
                    break
        
        return stats
