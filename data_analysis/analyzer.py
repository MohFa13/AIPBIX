import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Comprehensive data analyzer for Power BI dashboard generation
    """
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the dataset
        """
        # Convert data types if needed
        df = self._optimize_dtypes(df)
        
        analysis_results = {
            'basic_stats': self._get_basic_statistics(df),
            'column_analysis': self._analyze_columns(df),
            'data_quality': self._assess_data_quality(df),
            'correlations': self._calculate_correlations(df),
            'patterns': self._detect_patterns(df),
            'insights': self._generate_insights(df),
            'anomalies': self._detect_anomalies(df),
            'time_series_potential': self._assess_time_series_potential(df)
        }
        
        return analysis_results
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better analysis"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            # Try to convert potential datetime columns
            if df_optimized[col].dtype == 'object':
                if self._looks_like_datetime(df_optimized[col]):
                    try:
                        df_optimized[col] = pd.to_datetime(df_optimized[col], errors='coerce')
                    except:
                        pass
                
                # Try to convert to numeric if possible
                elif self._looks_like_numeric(df_optimized[col]):
                    try:
                        df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce')
                    except:
                        pass
        
        return df_optimized
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Check if a series looks like datetime"""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(10).astype(str)
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        for pattern in datetime_patterns:
            if any(pd.Series(sample).str.contains(pattern, na=False)):
                return True
        
        return False
    
    def _looks_like_numeric(self, series: pd.Series) -> bool:
        """Check if a series looks like numeric"""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(100)
        try:
            pd.to_numeric(sample)
            return True
        except:
            return False
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_df.columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values_total': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        if len(numeric_df.columns) > 0:
            stats.update({
                'numeric_summary': numeric_df.describe().to_dict(),
                'numeric_ranges': {col: {'min': numeric_df[col].min(), 'max': numeric_df[col].max()} 
                                 for col in numeric_df.columns}
            })
        
        return stats
    
    def _analyze_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze individual columns"""
        analysis_data = []
        
        for col in df.columns:
            col_data = df[col]
            
            analysis = {
                'column_name': col,
                'data_type': str(col_data.dtype),
                'non_null_count': col_data.notna().sum(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': col_data.isnull().sum() / len(col_data) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': col_data.nunique() / len(col_data) * 100,
                'is_constant': col_data.nunique() <= 1,
                'potential_key': col_data.nunique() == len(col_data) and col_data.notna().sum() == len(col_data)
            }
            
            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    'min_value': col_data.min(),
                    'max_value': col_data.max(),
                    'mean_value': col_data.mean(),
                    'median_value': col_data.median(),
                    'std_dev': col_data.std(),
                    'has_outliers': self._has_outliers(col_data),
                    'distribution_skew': col_data.skew() if col_data.std() > 0 else 0
                })
            
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                analysis.update({
                    'min_date': col_data.min(),
                    'max_date': col_data.max(),
                    'date_range_days': (col_data.max() - col_data.min()).days if pd.notna(col_data.min()) else 0
                })
            
            else:  # Categorical
                top_values = col_data.value_counts().head(5)
                analysis.update({
                    'top_values': top_values.to_dict(),
                    'most_frequent': top_values.index[0] if len(top_values) > 0 else None,
                    'most_frequent_count': top_values.iloc[0] if len(top_values) > 0 else 0
                })
            
            analysis_data.append(analysis)
        
        return pd.DataFrame(analysis_data)
    
    def _has_outliers(self, series: pd.Series) -> bool:
        """Detect outliers using IQR method"""
        if series.dtype not in ['int64', 'float64'] or len(series) < 10:
            return False
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers) > 0
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess overall data quality"""
        total_cells = len(df) * len(df.columns)
        
        # Completeness: percentage of non-null values
        completeness = df.notna().sum().sum() / total_cells
        
        # Uniqueness: average uniqueness across columns
        uniqueness_scores = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df[col]) if len(df[col]) > 0 else 0
            uniqueness_scores.append(min(unique_ratio, 1.0))  # Cap at 1.0
        uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0
        
        # Consistency: check for data type consistency
        consistency_issues = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    types = set(type(x).__name__ for x in sample)
                    if len(types) > 1:
                        consistency_issues += 1
        
        consistency = 1 - (consistency_issues / len(df.columns))
        
        # Validity: check for reasonable values
        validity_issues = 0
        total_numeric_cols = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            total_numeric_cols += 1
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Check for extreme outliers (beyond 3 standard deviations)
                if col_data.std() > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    extreme_outliers = np.sum(z_scores > 3)
                    if extreme_outliers > len(col_data) * 0.05:  # More than 5%
                        validity_issues += 1
        
        validity = 1 - (validity_issues / max(total_numeric_cols, 1))
        
        return {
            'completeness': completeness,
            'uniqueness': uniqueness,
            'consistency': consistency,
            'validity': validity,
            'overall_score': (completeness + uniqueness + consistency + validity) / 4
        }
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Strong Positive' if corr_value > 0 else 'Strong Negative'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'correlation_summary': {
                'total_pairs': len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) // 2,
                'strong_correlations_count': len(strong_correlations)
            }
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect interesting patterns in the data"""
        patterns = []
        
        # Time-based patterns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            for date_col in datetime_cols:
                patterns.append({
                    'type': 'temporal',
                    'description': f'Time series data available in {date_col}',
                    'column': date_col,
                    'potential_analysis': ['Trend analysis', 'Seasonality detection', 'Time-based aggregations']
                })
        
        # Categorical distribution patterns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for cat_col in categorical_cols:
            value_counts = df[cat_col].value_counts()
            if len(value_counts) > 1:
                # Check for skewed distribution
                top_value_percentage = value_counts.iloc[0] / len(df) * 100
                if top_value_percentage > 50:
                    patterns.append({
                        'type': 'distribution_skew',
                        'description': f'{cat_col} is heavily skewed towards "{value_counts.index[0]}" ({top_value_percentage:.1f}%)',
                        'column': cat_col,
                        'dominant_value': value_counts.index[0]
                    })
        
        # Numeric range patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for num_col in numeric_cols:
            col_data = df[num_col].dropna()
            if len(col_data) > 0:
                # Check for specific ranges or scales
                if col_data.min() >= 0 and col_data.max() <= 100:
                    patterns.append({
                        'type': 'percentage_scale',
                        'description': f'{num_col} appears to be on a percentage scale (0-100)',
                        'column': num_col,
                        'potential_format': 'percentage'
                    })
                elif col_data.min() >= 0 and col_data.max() > 1000:
                    patterns.append({
                        'type': 'large_numbers',
                        'description': f'{num_col} contains large numbers, may benefit from formatting',
                        'column': num_col,
                        'suggested_format': 'thousands/millions'
                    })
        
        return patterns
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from the data"""
        insights = []
        
        # Data volume insights
        if len(df) > 100000:
            insights.append("Large dataset detected - consider data aggregation for better dashboard performance")
        elif len(df) < 100:
            insights.append("Small dataset - all data points can be displayed without aggregation")
        
        # Missing data insights
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_percentage > 20:
            insights.append(f"High missing data rate ({missing_percentage:.1f}%) - consider data quality improvements")
        elif missing_percentage < 5:
            insights.append("Excellent data completeness - suitable for detailed analysis")
        
        # Column type insights
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
        
        if datetime_cols > 0:
            insights.append("Time series analysis opportunities available with date/time columns")
        
        if numeric_cols > categorical_cols:
            insights.append("Numeric-heavy dataset - excellent for statistical analysis and trend visualization")
        elif categorical_cols > numeric_cols:
            insights.append("Category-rich dataset - ideal for segmentation and classification analysis")
        
        # Uniqueness insights
        high_cardinality_cols = []
        potential_keys = []
        
        for col in df.columns:
            unique_percentage = df[col].nunique() / len(df) * 100
            if unique_percentage > 95:
                potential_keys.append(col)
            elif unique_percentage > 50:
                high_cardinality_cols.append(col)
        
        if potential_keys:
            insights.append(f"Potential key fields identified: {', '.join(potential_keys[:3])}")
        
        if high_cardinality_cols:
            insights.append(f"High-cardinality fields may need grouping: {', '.join(high_cardinality_cols[:3])}")
        
        # Correlation insights
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                insights.append(f"Strong correlations detected - consider multicollinearity in analysis")
        
        return insights
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect anomalies in numeric columns"""
        anomalies = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            # Using IQR method for anomaly detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            if len(outliers) > 0:
                anomalies[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(col_data) * 100,
                    'extreme_values': outliers.tolist()[:10]  # Top 10 extreme values
                }
        
        return anomalies
    
    def _assess_time_series_potential(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess potential for time series analysis"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return {'potential': False, 'reason': 'No datetime columns found'}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'potential': False, 'reason': 'No numeric columns for time series analysis'}
        
        # Analyze the main datetime column
        main_date_col = datetime_cols[0]
        date_data = df[main_date_col].dropna().sort_values()
        
        if len(date_data) < 10:
            return {'potential': False, 'reason': 'Insufficient datetime data points'}
        
        # Check for regular intervals
        date_diffs = date_data.diff().dropna()
        most_common_interval = date_diffs.mode()
        
        analysis = {
            'potential': True,
            'main_date_column': main_date_col,
            'date_range': {
                'start': date_data.min(),
                'end': date_data.max(),
                'span_days': (date_data.max() - date_data.min()).days
            },
            'data_points': len(date_data),
            'suggested_visualizations': ['Line charts', 'Area charts', 'Time-based bar charts'],
            'time_intelligence_opportunities': ['YTD calculations', 'Period comparisons', 'Trend analysis']
        }
        
        if len(most_common_interval) > 0:
            interval = most_common_interval.iloc[0]
            if interval.days == 1:
                analysis['likely_frequency'] = 'Daily'
            elif interval.days == 7:
                analysis['likely_frequency'] = 'Weekly'
            elif interval.days >= 28 and interval.days <= 31:
                analysis['likely_frequency'] = 'Monthly'
            elif interval.days >= 365:
                analysis['likely_frequency'] = 'Yearly'
            else:
                analysis['likely_frequency'] = 'Irregular'
        
        return analysis
