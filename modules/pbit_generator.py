import json
import zipfile
import io
import struct
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import base64
import uuid

class PBITGenerator:
    """Generates Power BI Template (.pbit) and Power BI (.pbix) files"""
    
    def __init__(self):
        self.visualization_counter = 0
    
    def generate_pbit(self, dataset: pd.DataFrame, data_analysis: Dict, design_prompt: str,
                     design_analysis: Dict, color_scheme: str = "Blue", 
                     include_kpis: bool = True, include_filters: bool = True) -> bytes:
        """Generate a complete PBIT file"""
        
        # Generate data model schema
        data_model_schema = self._create_data_model_schema(dataset, data_analysis)
        
        # Generate report layout
        report_layout = self._create_report_layout(dataset, data_analysis, design_analysis, 
                                                 color_scheme, include_kpis, include_filters)
        
        # Create DataMashup (Power Query)
        data_mashup = self._create_data_mashup(dataset)
        
        # Package into PBIT file
        pbit_content = self._package_pbit(data_model_schema, report_layout, data_mashup)
        
        return pbit_content
    
    def _create_data_model_schema(self, dataset: pd.DataFrame, data_analysis: Dict) -> Dict:
        """Create the DataModelSchema JSON structure"""
        
        # Create columns schema
        columns = []
        for col_name, dtype in dataset.dtypes.items():
            column_schema = {
                "name": str(col_name),
                "dataType": self._map_pandas_to_powerbi_type(dtype),
                "isHidden": False,
                "sortByColumn": None,
                "summarizeBy": "none" if pd.api.types.is_numeric_dtype(dtype) else "none",
                "formatString": self._get_format_string(dtype)
            }
            
            # Add specific properties for numeric columns
            if pd.api.types.is_numeric_dtype(dtype):
                column_schema["summarizeBy"] = "sum"
            
            columns.append(column_schema)
        
        # Create measures (basic KPIs)
        measures = self._create_default_measures(dataset)
        
        # Main table schema
        table_schema = {
            "name": "MainData",
            "columns": columns,
            "partitions": [
                {
                    "name": "MainData",
                    "source": {
                        "type": "m",
                        "expression": [
                            "let",
                            "    Source = #\"CSV_Data\"",
                            "in",
                            "    Source"
                        ]
                    }
                }
            ],
            "measures": measures,
            "isHidden": False
        }
        
        # Complete data model schema
        data_model = {
            "name": "AIGeneratedModel",
            "compatibilityLevel": 1550,
            "model": {
                "culture": "en-US",
                "dataSources": [
                    {
                        "type": "structured",
                        "name": "CSV_Data",
                        "connectionDetails": {
                            "protocol": "file",
                            "kind": "Local"
                        }
                    }
                ],
                "tables": [table_schema],
                "relationships": [],
                "roles": [],
                "expressions": [
                    {
                        "name": "CSV_Data",
                        "expression": [
                            "let",
                            "    Source = Csv.Document(File.Contents(\"[REPLACE_WITH_DATA_PATH]\"),[Delimiter=\",\", Columns=" + str(len(dataset.columns)) + ", Encoding=65001, QuoteStyle=QuoteStyle.None]),",
                            "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                            "    #\"Changed Type\" = Table.TransformColumnTypes(#\"Promoted Headers\",{" + self._generate_column_types(dataset) + "})",
                            "in",
                            "    #\"Changed Type\""
                        ]
                    }
                ]
            }
        }
        
        return data_model
    
    def _create_report_layout(self, dataset: pd.DataFrame, data_analysis: Dict, 
                            design_analysis: Dict, color_scheme: str, 
                            include_kpis: bool, include_filters: bool) -> Dict:
        """Create the report layout JSON structure"""
        
        # Initialize layout
        layout = {
            "version": "5.0",
            "id": str(uuid.uuid4()),
            "title": "AI Generated Dashboard",
            "pages": []
        }
        
        # Create main page
        page = self._create_page(dataset, data_analysis, design_analysis, 
                               color_scheme, include_kpis, include_filters)
        layout["pages"].append(page)
        
        return layout
    
    def _create_page(self, dataset: pd.DataFrame, data_analysis: Dict, 
                    design_analysis: Dict, color_scheme: str, 
                    include_kpis: bool, include_filters: bool) -> Dict:
        """Create a dashboard page"""
        
        page = {
            "id": str(uuid.uuid4()),
            "name": "Dashboard",
            "displayName": "Dashboard",
            "filters": [],
            "visuals": [],
            "width": 1280,
            "height": 720
        }
        
        current_y = 10
        
        # Add KPI cards if requested
        if include_kpis:
            kpi_visuals, current_y = self._create_kpi_cards(dataset, current_y, color_scheme)
            page["visuals"].extend(kpi_visuals)
        
        # Add filters if requested
        if include_filters:
            filter_visuals, current_y = self._create_filter_slicers(dataset, current_y)
            page["visuals"].extend(filter_visuals)
        
        # Add main visualizations
        main_visuals = self._create_main_visualizations(dataset, data_analysis, current_y, color_scheme)
        page["visuals"].extend(main_visuals)
        
        return page
    
    def _create_kpi_cards(self, dataset: pd.DataFrame, start_y: int, color_scheme: str) -> tuple:
        """Create KPI card visualizations"""
        kpi_visuals = []
        numeric_cols = dataset.select_dtypes(include=['number']).columns.tolist()
        
        card_width = 200
        card_height = 100
        cards_per_row = 4
        current_y = start_y
        
        # Create up to 4 KPI cards
        for i, col in enumerate(numeric_cols[:4]):
            x_pos = (i % cards_per_row) * (card_width + 20) + 20
            if i > 0 and i % cards_per_row == 0:
                current_y += card_height + 20
            
            kpi_visual = {
                "id": str(uuid.uuid4()),
                "type": "card",
                "title": f"Total {col.replace('_', ' ').title()}",
                "x": x_pos,
                "y": current_y,
                "width": card_width,
                "height": card_height,
                "config": {
                    "singleVisual": {
                        "visualType": "card",
                        "projections": {
                            "Values": [
                                {
                                    "queryRef": f"MainData.{col}",
                                    "active": True
                                }
                            ]
                        },
                        "prototypeQuery": {
                            "Version": 2,
                            "From": [
                                {
                                    "Name": "MainData",
                                    "Entity": "MainData"
                                }
                            ],
                            "Select": [
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {
                                                    "SourceRef": {
                                                        "Source": "MainData"
                                                    }
                                                },
                                                "Property": col
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": f"Sum of {col}"
                                }
                            ]
                        }
                    }
                }
            }
            kpi_visuals.append(kpi_visual)
        
        return kpi_visuals, current_y + card_height + 30
    
    def _create_filter_slicers(self, dataset: pd.DataFrame, start_y: int) -> tuple:
        """Create filter slicer visualizations"""
        filter_visuals = []
        text_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        
        slicer_width = 200
        slicer_height = 200
        current_y = start_y
        
        # Create up to 2 slicers for categorical columns
        for i, col in enumerate(text_cols[:2]):
            if dataset[col].nunique() <= 20:  # Only create slicers for columns with reasonable number of unique values
                x_pos = i * (slicer_width + 20) + 20
                
                slicer_visual = {
                    "id": str(uuid.uuid4()),
                    "type": "slicer",
                    "title": col.replace('_', ' ').title(),
                    "x": x_pos,
                    "y": current_y,
                    "width": slicer_width,
                    "height": slicer_height,
                    "config": {
                        "singleVisual": {
                            "visualType": "slicer",
                            "projections": {
                                "Values": [
                                    {
                                        "queryRef": f"MainData.{col}",
                                        "active": True
                                    }
                                ]
                            },
                            "prototypeQuery": {
                                "Version": 2,
                                "From": [
                                    {
                                        "Name": "MainData",
                                        "Entity": "MainData"
                                    }
                                ],
                                "Select": [
                                    {
                                        "Column": {
                                            "Expression": {
                                                "SourceRef": {
                                                    "Source": "MainData"
                                                }
                                            },
                                            "Property": col
                                        },
                                        "Name": col
                                    }
                                ]
                            }
                        }
                    }
                }
                filter_visuals.append(slicer_visual)
        
        return filter_visuals, current_y + slicer_height + 30 if filter_visuals else start_y
    
    def _create_main_visualizations(self, dataset: pd.DataFrame, data_analysis: Dict, 
                                  start_y: int, color_scheme: str) -> List[Dict]:
        """Create main dashboard visualizations"""
        visuals = []
        numeric_cols = dataset.select_dtypes(include=['number']).columns.tolist()
        text_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        date_cols = dataset.select_dtypes(include=['datetime64']).columns.tolist()
        
        visual_width = 400
        visual_height = 300
        visuals_per_row = 2
        current_y = start_y
        
        viz_count = 0
        
        # Create bar chart if we have categorical and numeric data
        if text_cols and numeric_cols:
            x_pos = (viz_count % visuals_per_row) * (visual_width + 40) + 40
            if viz_count > 0 and viz_count % visuals_per_row == 0:
                current_y += visual_height + 40
            
            bar_chart = self._create_bar_chart(text_cols[0], numeric_cols[0], x_pos, current_y, visual_width, visual_height)
            visuals.append(bar_chart)
            viz_count += 1
        
        # Create line chart if we have date and numeric data
        if date_cols and numeric_cols:
            x_pos = (viz_count % visuals_per_row) * (visual_width + 40) + 40
            if viz_count > 0 and viz_count % visuals_per_row == 0:
                current_y += visual_height + 40
            
            line_chart = self._create_line_chart(date_cols[0], numeric_cols[0], x_pos, current_y, visual_width, visual_height)
            visuals.append(line_chart)
            viz_count += 1
        
        # Create pie chart for categorical data
        if text_cols and numeric_cols and len(text_cols) >= 1:
            suitable_col = None
            for col in text_cols:
                if dataset[col].nunique() <= 10:  # Good for pie chart
                    suitable_col = col
                    break
            
            if suitable_col:
                x_pos = (viz_count % visuals_per_row) * (visual_width + 40) + 40
                if viz_count > 0 and viz_count % visuals_per_row == 0:
                    current_y += visual_height + 40
                
                pie_chart = self._create_pie_chart(suitable_col, numeric_cols[0], x_pos, current_y, visual_width, visual_height)
                visuals.append(pie_chart)
                viz_count += 1
        
        # Create table with top data
        if len(dataset.columns) > 0:
            x_pos = (viz_count % visuals_per_row) * (visual_width + 40) + 40
            if viz_count > 0 and viz_count % visuals_per_row == 0:
                current_y += visual_height + 40
            
            table_visual = self._create_table(dataset.columns.tolist()[:5], x_pos, current_y, visual_width, visual_height)
            visuals.append(table_visual)
            viz_count += 1
        
        return visuals
    
    def _create_bar_chart(self, category_col: str, value_col: str, x: int, y: int, width: int, height: int) -> Dict:
        """Create a bar chart visual"""
        return {
            "id": str(uuid.uuid4()),
            "type": "columnChart",
            "title": f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "config": {
                "singleVisual": {
                    "visualType": "columnChart",
                    "projections": {
                        "Category": [
                            {
                                "queryRef": f"MainData.{category_col}",
                                "active": True
                            }
                        ],
                        "Y": [
                            {
                                "queryRef": f"MainData.{value_col}",
                                "active": True
                            }
                        ]
                    },
                    "prototypeQuery": {
                        "Version": 2,
                        "From": [
                            {
                                "Name": "MainData",
                                "Entity": "MainData"
                            }
                        ],
                        "Select": [
                            {
                                "Column": {
                                    "Expression": {
                                        "SourceRef": {
                                            "Source": "MainData"
                                        }
                                    },
                                    "Property": category_col
                                },
                                "Name": category_col
                            },
                            {
                                "Aggregation": {
                                    "Expression": {
                                        "Column": {
                                            "Expression": {
                                                "SourceRef": {
                                                    "Source": "MainData"
                                                }
                                            },
                                            "Property": value_col
                                        }
                                    },
                                    "Function": 1
                                },
                                "Name": f"Sum of {value_col}"
                            }
                        ]
                    }
                }
            }
        }
    
    def _create_line_chart(self, date_col: str, value_col: str, x: int, y: int, width: int, height: int) -> Dict:
        """Create a line chart visual"""
        return {
            "id": str(uuid.uuid4()),
            "type": "lineChart",
            "title": f"{value_col.replace('_', ' ').title()} over {date_col.replace('_', ' ').title()}",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "config": {
                "singleVisual": {
                    "visualType": "lineChart",
                    "projections": {
                        "Category": [
                            {
                                "queryRef": f"MainData.{date_col}",
                                "active": True
                            }
                        ],
                        "Y": [
                            {
                                "queryRef": f"MainData.{value_col}",
                                "active": True
                            }
                        ]
                    }
                }
            }
        }
    
    def _create_pie_chart(self, category_col: str, value_col: str, x: int, y: int, width: int, height: int) -> Dict:
        """Create a pie chart visual"""
        return {
            "id": str(uuid.uuid4()),
            "type": "pieChart",
            "title": f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "config": {
                "singleVisual": {
                    "visualType": "pieChart",
                    "projections": {
                        "Category": [
                            {
                                "queryRef": f"MainData.{category_col}",
                                "active": True
                            }
                        ],
                        "Y": [
                            {
                                "queryRef": f"MainData.{value_col}",
                                "active": True
                            }
                        ]
                    }
                }
            }
        }
    
    def _create_table(self, columns: List[str], x: int, y: int, width: int, height: int) -> Dict:
        """Create a table visual"""
        values = []
        for col in columns:
            values.append({
                "queryRef": f"MainData.{col}",
                "active": True
            })
        
        return {
            "id": str(uuid.uuid4()),
            "type": "tableEx",
            "title": "Data Table",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "config": {
                "singleVisual": {
                    "visualType": "tableEx",
                    "projections": {
                        "Values": values
                    }
                }
            }
        }
    
    def _create_data_mashup(self, dataset: pd.DataFrame) -> bytes:
        """Create DataMashup binary file (MS-QDEFF format)"""
        # Create minimal DataMashup structure
        # This is a simplified version - full MS-QDEFF implementation would be much more complex
        
        mashup_data = {
            "Version": "1.0",
            "Sources": [
                {
                    "Name": "CSV_Data",
                    "Kind": "File",
                    "Path": "[PLACEHOLDER_PATH]"
                }
            ]
        }
        
        mashup_json = json.dumps(mashup_data).encode('utf-16le')
        
        # Create MS-QDEFF format structure
        output = io.BytesIO()
        
        # Version (4 bytes)
        output.write(struct.pack('<I', 1))
        
        # Package parts length (4 bytes) + data
        pkg_parts = b"Package parts placeholder"
        output.write(struct.pack('<I', len(pkg_parts)))
        output.write(pkg_parts)
        
        # Permissions length (4 bytes) + data
        permissions = b"Permissions placeholder"
        output.write(struct.pack('<I', len(permissions)))
        output.write(permissions)
        
        # Metadata length (4 bytes) + data
        output.write(struct.pack('<I', len(mashup_json)))
        output.write(mashup_json)
        
        return output.getvalue()
    
    def _package_pbit(self, data_model_schema: Dict, report_layout: Dict, data_mashup: bytes) -> bytes:
        """Package components into PBIT file"""
        
        pbit_buffer = io.BytesIO()
        
        with zipfile.ZipFile(pbit_buffer, 'w', zipfile.ZIP_DEFLATED) as pbit_zip:
            # Add DataModelSchema
            schema_json = json.dumps(data_model_schema, indent=2)
            pbit_zip.writestr('DataModelSchema', schema_json.encode('utf-8'))
            
            # Add Report Layout
            layout_json = json.dumps(report_layout, indent=2)
            pbit_zip.writestr('Report/Layout', layout_json.encode('utf-16le'))
            
            # Add DataMashup
            pbit_zip.writestr('DataMashup', data_mashup)
            
            # Add Content Types
            content_types = '''<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="json" ContentType="application/json"/>
</Types>'''
            pbit_zip.writestr('[Content_Types].xml', content_types.encode('utf-8'))
            
            # Add Security Bindings (empty)
            pbit_zip.writestr('SecurityBindings', b'')
            
            # Add Metadata
            metadata = {
                "version": "5.0",
                "createdDateTime": datetime.now().isoformat(),
                "modifiedDateTime": datetime.now().isoformat()
            }
            pbit_zip.writestr('Metadata', json.dumps(metadata).encode('utf-8'))
        
        pbit_buffer.seek(0)
        return pbit_buffer.read()
    
    def create_pbix_from_pbit(self, pbit_content: bytes, dataset: pd.DataFrame) -> bytes:
        """Create PBIX file from PBIT template by embedding data"""
        
        pbix_buffer = io.BytesIO()
        
        # Read PBIT content
        with zipfile.ZipFile(io.BytesIO(pbit_content), 'r') as pbit_zip:
            pbit_files = {name: pbit_zip.read(name) for name in pbit_zip.namelist()}
        
        # Create PBIX with embedded data
        with zipfile.ZipFile(pbix_buffer, 'w', zipfile.ZIP_DEFLATED) as pbix_zip:
            
            # Copy all PBIT files
            for filename, content in pbit_files.items():
                pbix_zip.writestr(filename, content)
            
            # Add embedded dataset (simplified approach)
            # In a real implementation, this would need to be properly formatted
            # for Power BI's internal data storage format
            
            # Convert dataset to CSV for embedding
            csv_buffer = io.StringIO()
            dataset.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            
            # Add dataset as embedded resource
            pbix_zip.writestr('EmbeddedData/MainData.csv', csv_data)
            
            # Update metadata to indicate embedded data
            metadata = {
                "version": "5.0",
                "createdDateTime": datetime.now().isoformat(),
                "modifiedDateTime": datetime.now().isoformat(),
                "hasEmbeddedData": True
            }
            pbix_zip.writestr('Metadata', json.dumps(metadata).encode('utf-8'))
        
        pbix_buffer.seek(0)
        return pbix_buffer.read()
    
    def _map_pandas_to_powerbi_type(self, pandas_dtype) -> str:
        """Map pandas data types to Power BI data types"""
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return "int64"
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return "double"
        elif pd.api.types.is_bool_dtype(pandas_dtype):
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return "dateTime"
        else:
            return "string"
    
    def _get_format_string(self, pandas_dtype) -> str:
        """Get format string for Power BI column"""
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return "0"
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return "0.00"
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return "mm/dd/yyyy"
        else:
            return ""
    
    def _generate_column_types(self, dataset: pd.DataFrame) -> str:
        """Generate Power Query column type transformations"""
        type_mappings = []
        
        for col_name, dtype in dataset.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                type_mappings.append(f'{{"{col_name}", Int64.Type}}')
            elif pd.api.types.is_float_dtype(dtype):
                type_mappings.append(f'{{"{col_name}", type number}}')
            elif pd.api.types.is_bool_dtype(dtype):
                type_mappings.append(f'{{"{col_name}", type logical}}')
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_mappings.append(f'{{"{col_name}", type datetime}}')
            else:
                type_mappings.append(f'{{"{col_name}", type text}}')
        
        return ", ".join(type_mappings)
    
    def _create_default_measures(self, dataset: pd.DataFrame) -> List[Dict]:
        """Create default DAX measures"""
        measures = []
        numeric_cols = dataset.select_dtypes(include=['number']).columns.tolist()
        
        # Create basic sum measures for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            measure = {
                "name": f"Total_{col}",
                "expression": f"SUM(MainData[{col}])",
                "formatString": "0.00" if pd.api.types.is_float_dtype(dataset[col]) else "0",
                "isHidden": False
            }
            measures.append(measure)
        
        # Add a count measure
        measures.append({
            "name": "Row_Count",
            "expression": "COUNTROWS(MainData)",
            "formatString": "0",
            "isHidden": False
        })
        
        return measures
