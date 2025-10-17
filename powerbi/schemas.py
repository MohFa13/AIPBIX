from typing import Dict, List, Any, Optional
import pandas as pd

class PowerBISchemas:
    """
    Schema definitions and templates for Power BI components
    """
    
    @staticmethod
    def get_base_datamodel_template() -> Dict[str, Any]:
        """
        Get base DataModelSchema template
        """
        return {
            "name": "Model",
            "compatibilityLevel": 1550,
            "model": {
                "culture": "en-US",
                "defaultPowerBIDataSourceVersion": "powerBI_V3",
                "sourceQueryCulture": "en-US",
                "tables": [],
                "relationships": [],
                "roles": [],
                "expressions": [],
                "annotations": [
                    {
                        "name": "ClientCompatibilityLevel",
                        "value": "700"
                    },
                    {
                        "name": "__PBI_TimeIntelligenceEnabled",
                        "value": "1"
                    }
                ],
                "perspectives": []
            }
        }
    
    @staticmethod
    def create_table_schema(table_name: str, columns: List[Dict[str, Any]], 
                           measures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create table schema for DataModelSchema
        """
        table_schema = {
            "name": table_name,
            "columns": columns,
            "partitions": [
                {
                    "name": f"{table_name}-Partition",
                    "source": {
                        "type": "m",
                        "expression": f"let\n    Source = #{{\"{table_name}\"}}\nin\n    Source"
                    }
                }
            ]
        }
        
        if measures:
            table_schema["measures"] = measures
        
        return table_schema
    
    @staticmethod
    def create_column_schema(column_name: str, data_type: str, 
                           source_column: Optional[str] = None,
                           is_key: bool = False, is_hidden: bool = False,
                           summarize_by: str = "none") -> Dict[str, Any]:
        """
        Create column schema for table definition
        """
        column_schema = {
            "name": column_name,
            "dataType": data_type,
            "sourceColumn": source_column or column_name,
            "summarizeBy": summarize_by,
            "isHidden": is_hidden,
            "isKey": is_key
        }
        
        # Add format string for specific data types
        if data_type == "dateTime":
            column_schema["formatString"] = "General Date"
        elif data_type == "double" and "percentage" in column_name.lower():
            column_schema["formatString"] = "0.00%"
        elif data_type in ["int64", "double"] and any(term in column_name.lower() 
                                                      for term in ["amount", "revenue", "sales", "cost", "price"]):
            column_schema["formatString"] = "$#,0.00"
        
        return column_schema
    
    @staticmethod
    def create_measure_schema(name: str, expression: str, 
                            format_string: Optional[str] = None,
                            description: Optional[str] = None,
                            display_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Create measure schema for table definition
        """
        measure_schema = {
            "name": name,
            "expression": expression
        }
        
        if format_string:
            measure_schema["formatString"] = format_string
        
        if description:
            measure_schema["description"] = description
        
        if display_folder:
            measure_schema["displayFolder"] = display_folder
        
        return measure_schema
    
    @staticmethod
    def infer_datatype_from_pandas(pandas_dtype: str) -> str:
        """
        Map pandas dtype to Power BI data type
        """
        dtype_mapping = {
            'int8': 'int64',
            'int16': 'int64', 
            'int32': 'int64',
            'int64': 'int64',
            'uint8': 'int64',
            'uint16': 'int64',
            'uint32': 'int64',
            'uint64': 'int64',
            'float16': 'double',
            'float32': 'double',
            'float64': 'double',
            'bool': 'boolean',
            'object': 'string',
            'string': 'string',
            'category': 'string',
            'datetime64[ns]': 'dateTime',
            'datetime64[ns, UTC]': 'dateTime',
            'timedelta64[ns]': 'duration'
        }
        
        # Handle datetime with timezone
        if 'datetime64' in pandas_dtype:
            return 'dateTime'
        
        return dtype_mapping.get(pandas_dtype, 'string')
    
    @staticmethod
    def get_default_visual_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get default configurations for different visual types
        """
        return {
            "barChart": {
                "visualType": "barChart",
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "legend": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "categoryAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "valueAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}]
                }
            },
            "lineChart": {
                "visualType": "lineChart", 
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "legend": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "categoryAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "valueAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "dataPoint": [{"properties": {"showAllDataPoints": {"expr": {"Literal": {"Value": "true"}}}}}]
                }
            },
            "pieChart": {
                "visualType": "pieChart",
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "legend": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "dataLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}]
                }
            },
            "card": {
                "visualType": "card",
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "categoryLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
                    "dataLabels": [{"properties": {"fontSize": {"expr": {"Literal": {"Value": "24D"}}}}}]
                }
            },
            "table": {
                "visualType": "tableEx",
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "grid": [{"properties": {"gridVertical": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "columnHeaders": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}]
                }
            },
            "scatterChart": {
                "visualType": "scatterChart",
                "objects": {
                    "general": [{"properties": {"responsive": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "legend": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "categoryAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}],
                    "valueAxis": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}}}]
                }
            }
        }
    
    @staticmethod
    def create_relationship_schema(from_table: str, from_column: str,
                                 to_table: str, to_column: str,
                                 cross_filtering_behavior: str = "bothDirections") -> Dict[str, Any]:
        """
        Create relationship schema for DataModelSchema
        """
        return {
            "name": f"{from_table}_{to_table}",
            "fromTable": from_table,
            "fromColumn": from_column,
            "toTable": to_table,
            "toColumn": to_column,
            "crossFilteringBehavior": cross_filtering_behavior,
            "isActive": True,
            "relyOnReferentialIntegrity": False,
            "securityFilteringBehavior": "bothDirections"
        }
    
    @staticmethod
    def get_common_dax_measures() -> List[Dict[str, Any]]:
        """
        Get common DAX measures for typical business scenarios
        """
        return [
            {
                "name": "Total Records",
                "expression": "COUNTROWS(Data)",
                "formatString": "#,0",
                "description": "Count of all records in the dataset"
            },
            {
                "name": "Distinct Count",
                "expression": "DISTINCTCOUNT(Data[ID])",
                "formatString": "#,0", 
                "description": "Count of unique records"
            },
            {
                "name": "Previous Period",
                "expression": "CALCULATE([Total Sales], PREVIOUSMONTH(Data[Date]))",
                "formatString": "$#,0",
                "description": "Previous period comparison"
            },
            {
                "name": "Growth Rate",
                "expression": "DIVIDE([Total Sales] - [Previous Period], [Previous Period], 0)",
                "formatString": "0.00%",
                "description": "Period over period growth rate"
            },
            {
                "name": "Running Total",
                "expression": "CALCULATE([Total Sales], FILTER(ALLSELECTED(Data), Data[Date] <= MAX(Data[Date])))",
                "formatString": "$#,0",
                "description": "Running total calculation"
            }
        ]
    
    @staticmethod
    def get_time_intelligence_measures(date_column: str = "Date", 
                                     value_measure: str = "Total Sales") -> List[Dict[str, Any]]:
        """
        Generate time intelligence measures for date analysis
        """
        return [
            {
                "name": f"{value_measure} YTD",
                "expression": f"TOTALYTD([{value_measure}], Data[{date_column}])",
                "formatString": "$#,0",
                "description": f"Year to date {value_measure.lower()}"
            },
            {
                "name": f"{value_measure} QTD", 
                "expression": f"TOTALQTD([{value_measure}], Data[{date_column}])",
                "formatString": "$#,0",
                "description": f"Quarter to date {value_measure.lower()}"
            },
            {
                "name": f"{value_measure} MTD",
                "expression": f"TOTALMTD([{value_measure}], Data[{date_column}])",
                "formatString": "$#,0",
                "description": f"Month to date {value_measure.lower()}"
            },
            {
                "name": f"{value_measure} Previous Year",
                "expression": f"CALCULATE([{value_measure}], SAMEPERIODLASTYEAR(Data[{date_column}]))",
                "formatString": "$#,0",
                "description": f"Previous year {value_measure.lower()}"
            },
            {
                "name": f"{value_measure} YoY Growth",
                "expression": f"DIVIDE([{value_measure}] - [{value_measure} Previous Year], [{value_measure} Previous Year], 0)",
                "formatString": "0.00%",
                "description": f"Year over year growth for {value_measure.lower()}"
            }
        ]
