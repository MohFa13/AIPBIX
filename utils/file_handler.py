import pandas as pd
import io
from typing import Dict, List, Any, Tuple
import chardet
import warnings
warnings.filterwarnings('ignore')

class FileHandler:
    """
    Handles file loading, validation, and processing for various formats
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
        self.max_file_size_mb = 100
        self.max_rows = 1000000
    
    def load_file(self, uploaded_file) -> pd.DataFrame:
        """
        Load file and return pandas DataFrame
        """
        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.max_file_size_mb} MB)")
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(self.supported_formats)}")
        
        try:
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        """
        Load CSV file with encoding detection and delimiter inference
        """
        # Read raw bytes for encoding detection
        raw_data = uploaded_file.getvalue()
        
        # Detect encoding
        encoding_result = chardet.detect(raw_data[:10000])  # Sample first 10KB
        encoding = encoding_result.get('encoding', 'utf-8')
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        try:
            # Try to read with detected encoding
            df = pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to common encodings
            for fallback_encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=fallback_encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not determine file encoding")
        
        # Validate loaded data
        if df.empty:
            raise ValueError("CSV file is empty")
        
        if len(df) > self.max_rows:
            raise ValueError(f"File contains too many rows ({len(df)}). Maximum allowed: {self.max_rows}")
        
        return df
    
    def _load_excel(self, uploaded_file) -> pd.DataFrame:
        """
        Load Excel file and handle multiple sheets
        """
        try:
            # Read Excel file and get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 0:
                raise ValueError("Excel file contains no sheets")
            
            # Use first sheet by default
            # In a more advanced implementation, you could let users choose
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
            
            if df.empty:
                raise ValueError(f"Excel sheet '{sheet_names[0]}' is empty")
            
            if len(df) > self.max_rows:
                raise ValueError(f"Sheet contains too many rows ({len(df)}). Maximum allowed: {self.max_rows}")
            
            return df
            
        except Exception as e:
            if "Excel" in str(e):
                raise ValueError(f"Error reading Excel file: {str(e)}")
            else:
                raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data and return validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check for empty dataframe
        if df.empty:
            validation_results['errors'].append("Dataset is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check minimum data requirements
        if len(df) < 2:
            validation_results['errors'].append("Dataset must contain at least 2 rows")
            validation_results['is_valid'] = False
        
        if len(df.columns) < 1:
            validation_results['errors'].append("Dataset must contain at least 1 column")
            validation_results['is_valid'] = False
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicate_cols = [col for col in df.columns if list(df.columns).count(col) > 1]
            validation_results['warnings'].append(f"Duplicate column names found: {list(set(duplicate_cols))}")
        
        # Check for columns with all missing values
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
            validation_results['warnings'].append(f"Columns with all missing values: {empty_cols}")
        
        # Check for high missing data percentage
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 50:
            validation_results['warnings'].append(f"High missing data percentage: {missing_percentage:.1f}%")
        elif missing_percentage > 20:
            validation_results['info'].append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        # Check for columns with single unique value
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation_results['info'].append(f"Columns with constant values: {constant_cols}")
        
        # Check for potential data types that should be converted
        potential_numeric_cols = []
        potential_date_cols = []
        
        for col in df.select_dtypes(include=['object']).columns:
            # Check if object column could be numeric
            if self._is_potential_numeric(df[col]):
                potential_numeric_cols.append(col)
            
            # Check if object column could be datetime
            if self._is_potential_datetime(df[col]):
                potential_date_cols.append(col)
        
        if potential_numeric_cols:
            validation_results['info'].append(f"Potential numeric columns (currently text): {potential_numeric_cols}")
        
        if potential_date_cols:
            validation_results['info'].append(f"Potential date columns (currently text): {potential_date_cols}")
        
        # Check for reasonable data size
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_usage_mb > 50:
            validation_results['warnings'].append(f"Large dataset ({memory_usage_mb:.1f} MB) may impact performance")
        
        return validation_results
    
    def _is_potential_numeric(self, series: pd.Series, sample_size: int = 100) -> bool:
        """
        Check if a series could be converted to numeric
        """
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        # Try to convert sample to numeric
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_potential_datetime(self, series: pd.Series, sample_size: int = 100) -> bool:
        """
        Check if a series could be converted to datetime
        """
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        # Try to convert sample to datetime
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        summary = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum()
            },
            'column_types': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'datetime': len(datetime_cols),
                'other': len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)
            },
            'column_details': {
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols
            }
        }
        
        # Add statistical summary for numeric columns
        if numeric_cols:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Add value counts for categorical columns (top 5 for each)
        if categorical_cols:
            summary['categorical_summary'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = df[col].value_counts().head(5)
                summary['categorical_summary'][col] = value_counts.to_dict()
        
        return summary
    
    def suggest_data_improvements(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest improvements for data quality
        """
        suggestions = []
        
        # Missing data suggestions
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            high_missing_cols = [col for col in missing_cols if df[col].isnull().sum() / len(df) > 0.2]
            if high_missing_cols:
                suggestions.append(f"Consider removing or imputing high-missing columns: {high_missing_cols}")
            else:
                suggestions.append(f"Consider imputing missing values in: {missing_cols[:5]}")
        
        # Data type suggestions
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if self._is_potential_numeric(df[col]):
                suggestions.append(f"Consider converting '{col}' to numeric type")
            elif self._is_potential_datetime(df[col]):
                suggestions.append(f"Consider converting '{col}' to datetime type")
        
        # Duplicate data suggestions
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            suggestions.append(f"Consider removing {duplicate_count} duplicate rows")
        
        # High cardinality suggestions
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8 and df[col].nunique() > 10:
                suggestions.append(f"'{col}' has high cardinality - consider grouping similar values")
        
        # Performance suggestions
        if len(df) > 100000:
            suggestions.append("Consider data sampling or aggregation for large datasets to improve dashboard performance")
        
        return suggestions
