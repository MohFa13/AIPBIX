import json
import zipfile
import io
import pandas as pd
from typing import Dict, List, Any
import uuid
from datetime import datetime
import struct

class PBITGenerator:
    """
    Generator for Power BI Template (PBIT) files
    """
    
    def __init__(self):
        self.template_version = "2.0"
        self.compatibility_level = 1550
    
    def generate_pbit(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                      viz_suggestions: List[Dict], requirements: Dict[str, Any],
                      include_sample_data: bool = True) -> bytes:
        """
        Generate a complete PBIT file
        """
        # Generate DataModelSchema
        schema = self._create_datamodel_schema(data, analysis_results, requirements)
        
        # Generate Report Layout
        layout = self._create_report_layout(viz_suggestions, requirements, data.columns.tolist())
        
        # Generate DataMashup (M code for data connections)
        datamashup = self._create_datamashup(data, include_sample_data)
        
        # Create PBIT ZIP archive
        pbit_buffer = io.BytesIO()
        
        with zipfile.ZipFile(pbit_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as pbit:
            # Add DataModelSchema
            pbit.writestr('DataModelSchema', json.dumps(schema, indent=2))
            
            # Add Report Layout
            pbit.writestr('Report/Layout', json.dumps(layout))
            
            # Add DataMashup
            pbit.writestr('DataMashup', datamashup)
            
            # Add metadata files
            pbit.writestr('[Content_Types].xml', self._get_content_types_xml())
            pbit.writestr('SecurityBindings', b'')
            pbit.writestr('Settings', json.dumps(self._get_settings()))
            pbit.writestr('Version', self.template_version)
            
            # Add DiagramState (empty for now)
            pbit.writestr('DiagramState', json.dumps({"version": "1.1", "objects": []}))
            
            # Add ReportMetadata
            metadata = {
                "version": "3.0",
                "createdFrom": "AI Dashboard Agent",
                "createdDateTime": datetime.now().isoformat()
            }
            pbit.writestr('ReportMetadata', json.dumps(metadata))
        
        pbit_buffer.seek(0)
        return pbit_buffer.getvalue()
    
    def _create_datamodel_schema(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                                requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the DataModelSchema JSON structure
        """
        # Generate columns for the main table
        columns = []
        for col_name in data.columns:
            col_dtype = str(data[col_name].dtype)
            
            # Map pandas dtypes to Power BI data types
            if 'int' in col_dtype:
                pbi_type = "int64"
            elif 'float' in col_dtype:
                pbi_type = "double"
            elif 'datetime' in col_dtype:
                pbi_type = "dateTime"
            elif 'bool' in col_dtype:
                pbi_type = "boolean"
            else:
                pbi_type = "string"
            
            columns.append({
                "name": col_name,
                "dataType": pbi_type,
                "sourceColumn": col_name,
                "summarizeBy": "none" if pbi_type == "string" else "sum",
                "isHidden": False,
                "isKey": False,
                "sortByColumn": None
            })
        
        # Create measures based on analysis
        measures = self._create_measures(data, analysis_results, requirements)
        
        # Main table structure
        main_table = {
            "name": "Data",
            "columns": columns,
            "partitions": [
                {
                    "name": "Data-Partition",
                    "source": {
                        "type": "m",
                        "expression": "let\n    Source = #\"Data\"\nin\n    Source"
                    }
                }
            ],
            "measures": measures
        }
        
        # Complete schema structure
        schema = {
            "name": "AIGeneratedModel",
            "compatibilityLevel": self.compatibility_level,
            "model": {
                "culture": "en-US",
                "defaultPowerBIDataSourceVersion": "powerBI_V3",
                "sourceQueryCulture": "en-US",
                "tables": [main_table],
                "relationships": [],  # No relationships for single table
                "roles": [],
                "expressions": [
                    {
                        "name": "Data",
                        "kind": "m",
                        "expression": self._generate_m_expression(data)
                    }
                ],
                "annotations": [
                    {
                        "name": "PBI_QueryOrder",
                        "value": "[\"Data\"]"
                    },
                    {
                        "name": "ClientCompatibilityLevel",
                        "value": "700"
                    }
                ]
            }
        }
        
        return schema
    
    def _create_measures(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                        requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create DAX measures based on data analysis and requirements
        """
        measures = []
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # Create basic aggregate measures for numeric columns
        for col in numeric_columns:
            # Total measure
            measures.append({
                "name": f"Total {col}",
                "expression": f"SUM(Data[{col}])",
                "formatString": "#,0" if 'amount' in col.lower() or 'revenue' in col.lower() else "#,0.00",
                "description": f"Sum of {col}"
            })
            
            # Average measure
            measures.append({
                "name": f"Average {col}",
                "expression": f"AVERAGE(Data[{col}])",
                "formatString": "#,0.00",
                "description": f"Average of {col}"
            })
        
        # Create KPI measures based on requirements
        if 'kpis' in requirements:
            for kpi in requirements['kpis']:
                kpi_name = kpi.get('name', 'Custom KPI')
                calculation = kpi.get('calculation', 'sum')
                
                # Find relevant column for KPI
                relevant_col = None
                for col in numeric_columns:
                    if any(term in col.lower() for term in kpi_name.lower().split()):
                        relevant_col = col
                        break
                
                if relevant_col:
                    if calculation.lower() == 'sum':
                        expression = f"SUM(Data[{relevant_col}])"
                    elif calculation.lower() == 'average':
                        expression = f"AVERAGE(Data[{relevant_col}])"
                    elif calculation.lower() == 'count':
                        expression = f"COUNT(Data[{relevant_col}])"
                    elif calculation.lower() == 'max':
                        expression = f"MAX(Data[{relevant_col}])"
                    elif calculation.lower() == 'min':
                        expression = f"MIN(Data[{relevant_col}])"
                    else:
                        expression = f"SUM(Data[{relevant_col}])"
                    
                    format_string = "#,0"
                    if kpi.get('format') == 'currency':
                        format_string = "$#,0"
                    elif kpi.get('format') == 'percentage':
                        format_string = "0.00%"
                    
                    measures.append({
                        "name": kpi_name,
                        "expression": expression,
                        "formatString": format_string,
                        "description": f"KPI: {kpi_name}"
                    })
        
        # Add utility measures
        measures.extend([
            {
                "name": "Total Records",
                "expression": "COUNTROWS(Data)",
                "formatString": "#,0",
                "description": "Total number of records"
            }
        ])
        
        return measures
    
    def _generate_m_expression(self, data: pd.DataFrame) -> str:
        """
        Generate Power Query M expression for data source
        """
        # For a template, we create a parameterized connection
        # This is a simplified M expression that users can modify
        
        sample_data_rows = []
        for _, row in data.head(3).iterrows():
            row_values = []
            for col in data.columns:
                value = row[col]
                if pd.isna(value):
                    row_values.append("null")
                elif isinstance(value, str):
                    row_values.append(f'"{value}"')
                else:
                    row_values.append(str(value))
            sample_data_rows.append("{" + ", ".join(row_values) + "}")
        
        column_definitions = []
        for col in data.columns:
            dtype_mapping = {
                'int64': 'Int64.Type',
                'float64': 'Double.Type', 
                'object': 'Text.Type',
                'datetime64[ns]': 'DateTime.Type',
                'bool': 'Logical.Type'
            }
            
            pbi_type = dtype_mapping.get(str(data[col].dtype), 'Text.Type')
            column_definitions.append(f'{{"{col}", {pbi_type}}}')
        
        m_expression = f"""let
    Source = Excel.Workbook(File.Contents("C:\\YourDataFile.xlsx"), null, true),
    Sheet1_Sheet = Source{{[Item="Sheet1",Kind="Sheet"]}}[Data],
    #"Promoted Headers" = Table.PromoteHeaders(Sheet1_Sheet, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{
        {', '.join(column_definitions)}
    }})
in
    #"Changed Type\""""
        
        return m_expression
    
    def _create_report_layout(self, viz_suggestions: List[Dict], requirements: Dict[str, Any], 
                             columns: List[str]) -> Dict[str, Any]:
        """
        Create the report layout JSON with visualizations
        """
        # Calculate layout dimensions
        page_width = 1280
        page_height = 720
        viz_count = len(viz_suggestions)
        
        # Determine grid layout
        if viz_count <= 4:
            cols, rows = 2, 2
        elif viz_count <= 6:
            cols, rows = 3, 2
        elif viz_count <= 9:
            cols, rows = 3, 3
        else:
            cols, rows = 4, 3
            viz_suggestions = viz_suggestions[:12]  # Limit to 12 visualizations
        
        viz_width = page_width // cols - 20
        viz_height = page_height // rows - 40
        
        # Generate visualizations
        visuals = []
        for i, viz_suggestion in enumerate(viz_suggestions):
            row = i // cols
            col = i % cols
            
            x = col * (viz_width + 20) + 10
            y = row * (viz_height + 40) + 30
            
            visual = self._create_visual(viz_suggestion, x, y, viz_width, viz_height, columns)
            if visual:
                visuals.append(visual)
        
        # Create layout structure
        layout = {
            "id": 0,
            "resourcePackages": [
                {
                    "name": "SharedResources",
                    "items": [
                        {
                            "name": "BaseTheme",
                            "path": "BaseTheme.json"
                        }
                    ]
                }
            ],
            "config": json.dumps({
                "version": "5.43",
                "themeCollection": {
                    "baseTheme": {
                        "name": "CY24SU06",
                        "version": "5.43",
                        "type": 2
                    }
                },
                "activeSectionIndex": 0,
                "defaultDrillFilterOtherVisuals": True,
                "sections": [
                    {
                        "name": "ReportSection",
                        "displayName": "AI Generated Dashboard",
                        "visualContainers": visuals,
                        "width": page_width,
                        "height": page_height,
                        "displayOption": 1,
                        "filters": [],
                        "objects": {
                            "section": [
                                {
                                    "properties": {
                                        "verticalAlignment": {
                                            "expr": {
                                                "Literal": {
                                                    "Value": "\"Middle\""
                                                }
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            })
        }
        
        return layout
    
    def _create_visual(self, viz_suggestion: Dict, x: int, y: int, width: int, height: int, 
                      columns: List[str]) -> Dict[str, Any]:
        """
        Create a single visual configuration
        """
        viz_type = viz_suggestion.get('type', 'bar')
        viz_fields = viz_suggestion.get('fields', [])
        
        # Map visualization types to Power BI visual types
        visual_type_mapping = {
            'bar': 'barChart',
            'line': 'lineChart',
            'pie': 'pieChart',
            'scatter': 'scatterChart',
            'table': 'tableEx',
            'card': 'card'
        }
        
        pbi_visual_type = visual_type_mapping.get(viz_type, 'barChart')
        
        # Create data roles based on visualization type and available fields
        data_roles = []
        
        if viz_type == 'bar' and len(viz_fields) >= 2:
            categorical_field = viz_fields[0]
            numeric_field = viz_fields[1] if len(viz_fields) > 1 else viz_fields[0]
            
            data_roles = [
                {
                    "Category": [
                        {
                            "queryRef": f"Data.{categorical_field}"
                        }
                    ]
                },
                {
                    "Y": [
                        {
                            "queryRef": f"Sum(Data.{numeric_field})"
                        }
                    ]
                }
            ]
        
        elif viz_type == 'line' and len(viz_fields) >= 2:
            x_field = viz_fields[0]
            y_field = viz_fields[1] if len(viz_fields) > 1 else viz_fields[0]
            
            data_roles = [
                {
                    "Category": [
                        {
                            "queryRef": f"Data.{x_field}"
                        }
                    ]
                },
                {
                    "Y": [
                        {
                            "queryRef": f"Sum(Data.{y_field})"
                        }
                    ]
                }
            ]
        
        elif viz_type == 'pie' and len(viz_fields) >= 2:
            category_field = viz_fields[0]
            value_field = viz_fields[1] if len(viz_fields) > 1 else viz_fields[0]
            
            data_roles = [
                {
                    "Category": [
                        {
                            "queryRef": f"Data.{category_field}"
                        }
                    ]
                },
                {
                    "Y": [
                        {
                            "queryRef": f"Sum(Data.{value_field})"
                        }
                    ]
                }
            ]
        
        elif viz_type == 'card' and len(viz_fields) >= 1:
            value_field = viz_fields[0]
            data_roles = [
                {
                    "Y": [
                        {
                            "queryRef": f"Sum(Data.{value_field})"
                        }
                    ]
                }
            ]
        
        elif viz_type == 'table':
            data_roles = [
                {
                    "Values": [
                        {
                            "queryRef": f"Data.{field}"
                        }
                        for field in viz_fields[:6]  # Limit to 6 columns
                    ]
                }
            ]
        
        # Create the visual container
        visual_container = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "config": json.dumps({
                "name": str(uuid.uuid4()),
                "layouts": [
                    {
                        "id": 0,
                        "position": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        }
                    }
                ],
                "singleVisual": {
                    "visualType": pbi_visual_type,
                    "projections": data_roles,
                    "prototypeQuery": {
                        "Version": 2,
                        "From": [
                            {
                                "Name": "d",
                                "Entity": "Data"
                            }
                        ],
                        "Select": [
                            {
                                "Column": {
                                    "Expression": {
                                        "SourceRef": {
                                            "Source": "d"
                                        }
                                    },
                                    "Property": field
                                },
                                "Name": f"Data.{field}"
                            }
                            for field in viz_fields[:5]
                        ]
                    },
                    "objects": self._get_visual_formatting(viz_type)
                }
            }),
            "filters": "[]"
        }
        
        return visual_container
    
    def _get_visual_formatting(self, viz_type: str) -> Dict[str, Any]:
        """
        Get default formatting for visual types
        """
        formatting = {}
        
        if viz_type in ['bar', 'line']:
            formatting = {
                "general": [
                    {
                        "properties": {
                            "responsive": {
                                "expr": {
                                    "Literal": {
                                        "Value": "true"
                                    }
                                }
                            }
                        }
                    }
                ],
                "legend": [
                    {
                        "properties": {
                            "show": {
                                "expr": {
                                    "Literal": {
                                        "Value": "true"
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        
        return formatting
    
    def _create_datamashup(self, data: pd.DataFrame, include_sample_data: bool = True) -> bytes:
        """
        Create DataMashup binary file (simplified MS-QDEFF format)
        """
        # This is a simplified implementation of the MS-QDEFF format
        # In a production environment, you would need a more complete implementation
        
        # Create basic DataMashup structure
        datamashup_content = {
            "Version": "2.0",
            "Queries": [
                {
                    "Name": "Data",
                    "Expression": self._generate_m_expression(data)
                }
            ]
        }
        
        # Convert to simplified binary format
        json_str = json.dumps(datamashup_content)
        json_bytes = json_str.encode('utf-16le')
        
        # MS-QDEFF structure (simplified)
        header = struct.pack('<I', 1)  # Version
        pkg_parts_len = struct.pack('<I', 0)  # Package parts length
        perm_len = struct.pack('<I', 0)  # Permissions length  
        metadata_len = struct.pack('<I', len(json_bytes))  # Metadata length
        
        datamashup_binary = header + pkg_parts_len + perm_len + metadata_len + json_bytes
        
        return datamashup_binary
    
    def _get_content_types_xml(self) -> str:
        """
        Get Content Types XML for PBIT file
        """
        return '''<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="json" ContentType="application/json" />
    <Override PartName="/DataModelSchema" ContentType="application/json" />
    <Override PartName="/Report/Layout" ContentType="application/json" />
    <Override PartName="/DataMashup" ContentType="application/octet-stream" />
    <Override PartName="/SecurityBindings" ContentType="application/octet-stream" />
    <Override PartName="/Settings" ContentType="application/json" />
    <Override PartName="/DiagramState" ContentType="application/json" />
    <Override PartName="/ReportMetadata" ContentType="application/json" />
</Types>'''
    
    def _get_settings(self) -> Dict[str, Any]:
        """
        Get default settings for PBIT file
        """
        return {
            "version": "1.0",
            "settings": {
                "useEnhancedInferenceOptions": True,
                "allowNativeQueries": False,
                "enableDatasetAutoDiscovery": True
            }
        }
    
    def convert_to_pbix(self, pbit_content: bytes, data: pd.DataFrame) -> bytes:
        """
        Convert PBIT template to PBIX by including actual data
        Note: This is a simplified conversion for demonstration
        In production, full PBIX generation requires complex data serialization
        """
        # For this implementation, we'll create a PBIX that's essentially
        # a PBIT with embedded data references
        
        pbix_buffer = io.BytesIO()
        pbit_zip = zipfile.ZipFile(io.BytesIO(pbit_content), 'r')
        
        with zipfile.ZipFile(pbix_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as pbix:
            # Copy all files from PBIT
            for item in pbit_zip.infolist():
                content = pbit_zip.read(item.filename)
                pbix.writestr(item.filename, content)
            
            # Add data cache (simplified representation)
            # In a real implementation, this would be the compressed VertiPaq data
            data_cache = {
                "version": "1.0",
                "tables": {
                    "Data": {
                        "rowCount": len(data),
                        "columnCount": len(data.columns),
                        "lastRefresh": datetime.now().isoformat(),
                        "sampleData": data.head(100).to_dict('records') if len(data) > 0 else []
                    }
                }
            }
            
            pbix.writestr('DataCache', json.dumps(data_cache))
            
            # Update metadata to indicate this is a PBIX
            metadata = json.loads(pbit_zip.read('ReportMetadata'))
            metadata['type'] = 'PBIX'
            metadata['hasData'] = True
            pbix.writestr('ReportMetadata', json.dumps(metadata))
        
        pbit_zip.close()
        pbix_buffer.seek(0)
        return pbix_buffer.getvalue()
