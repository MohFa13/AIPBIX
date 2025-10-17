"""
Base PBIT template structures and utilities
"""

def get_base_data_model_schema():
    """Get the base structure for Power BI data model schema"""
    return {
        "name": "AIGeneratedModel",
        "compatibilityLevel": 1550,
        "model": {
            "culture": "en-US",
            "dataAccessOptions": {
                "legacyRedirects": True,
                "returnErrorValuesAsNull": True
            },
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "en-US",
            "tables": [],
            "relationships": [],
            "roles": [],
            "expressions": [],
            "perspectives": [],
            "queryGroups": []
        }
    }

def get_base_report_layout():
    """Get the base structure for Power BI report layout"""
    return {
        "version": "5.0",
        "authoringInfo": {
            "version": "5.0"
        },
        "config": "",
        "layoutOptimization": 0,
        "id": "00000000-0000-0000-0000-000000000000",
        "title": "",
        "pages": [],
        "resourcePackages": [],
        "settings": {
            "useStylableVisualContainerHeader": True
        }
    }

def get_content_types_xml():
    """Get the [Content_Types].xml content for PBIT files"""
    return '''<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="json" ContentType="application/json"/>
</Types>'''

def get_base_table_schema(table_name: str, columns: list):
    """Generate base table schema for Power BI"""
    return {
        "name": table_name,
        "lineageTag": f"{table_name}_tag",
        "columns": columns,
        "partitions": [
            {
                "name": f"{table_name}_partition",
                "mode": "import",
                "source": {
                    "type": "m",
                    "expression": [
                        "let",
                        f"    Source = #{table_name}_source",
                        "in",
                        "    Source"
                    ]
                }
            }
        ],
        "measures": [],
        "annotations": []
    }

def get_base_column_schema(column_name: str, data_type: str, format_string: str = ""):
    """Generate base column schema for Power BI"""
    column = {
        "name": column_name,
        "lineageTag": f"{column_name}_tag",
        "dataType": data_type,
        "sourceColumn": column_name,
        "summarizeBy": "none",
        "annotations": []
    }
    
    if format_string:
        column["formatString"] = format_string
    
    # Set appropriate summarizeBy for numeric columns
    if data_type in ["int64", "double", "decimal"]:
        column["summarizeBy"] = "sum"
    
    return column

def get_base_measure_schema(measure_name: str, expression: str, format_string: str = ""):
    """Generate base measure schema for Power BI"""
    measure = {
        "name": measure_name,
        "lineageTag": f"{measure_name}_tag",
        "expression": expression,
        "annotations": []
    }
    
    if format_string:
        measure["formatString"] = format_string
    
    return measure

def get_visualization_templates():
    """Get templates for common Power BI visualizations"""
    return {
        "card": {
            "singleVisual": {
                "visualType": "card",
                "projections": {
                    "Values": []
                },
                "vcObjects": {
                    "card": [
                        {
                            "properties": {
                                "fontSize": {
                                    "expr": {
                                        "Literal": {
                                            "Value": "14D"
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
        "columnChart": {
            "singleVisual": {
                "visualType": "columnChart",
                "projections": {
                    "Category": [],
                    "Y": []
                },
                "vcObjects": {
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
                    ]
                }
            }
        },
        "lineChart": {
            "singleVisual": {
                "visualType": "lineChart",
                "projections": {
                    "Category": [],
                    "Y": []
                },
                "vcObjects": {
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
                    ]
                }
            }
        },
        "pieChart": {
            "singleVisual": {
                "visualType": "pieChart",
                "projections": {
                    "Category": [],
                    "Y": []
                },
                "vcObjects": {
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
                    ]
                }
            }
        },
        "tableEx": {
            "singleVisual": {
                "visualType": "tableEx",
                "projections": {
                    "Values": []
                },
                "vcObjects": {
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
                    ]
                }
            }
        },
        "slicer": {
            "singleVisual": {
                "visualType": "slicer",
                "projections": {
                    "Values": []
                },
                "vcObjects": {
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
                    ]
                }
            }
        }
    }

def get_color_schemes():
    """Get predefined color schemes for Power BI dashboards"""
    return {
        "Blue": {
            "primary": "#1f77b4",
            "secondary": "#aec7e8",
            "accent": "#ff7f0e",
            "background": "#ffffff",
            "text": "#333333"
        },
        "Green": {
            "primary": "#2ca02c",
            "secondary": "#98df8a",
            "accent": "#d62728",
            "background": "#ffffff",
            "text": "#333333"
        },
        "Red": {
            "primary": "#d62728",
            "secondary": "#ff9896",
            "accent": "#2ca02c",
            "background": "#ffffff",
            "text": "#333333"
        },
        "Purple": {
            "primary": "#9467bd",
            "secondary": "#c5b0d5",
            "accent": "#ff7f0e",
            "background": "#ffffff",
            "text": "#333333"
        },
        "Orange": {
            "primary": "#ff7f0e",
            "secondary": "#ffbb78",
            "accent": "#1f77b4",
            "background": "#ffffff",
            "text": "#333333"
        }
    }

def get_power_query_template(table_name: str, column_count: int):
    """Generate Power Query M expression template"""
    return [
        "let",
        f"    Source = Csv.Document(File.Contents(\"[REPLACE_WITH_DATA_PATH]\"),[Delimiter=\",\", Columns={column_count}, Encoding=65001, QuoteStyle=QuoteStyle.None]),",
        "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
        "    #\"Changed Type\" = Table.TransformColumnTypes(#\"Promoted Headers\",{[COLUMN_TYPES]})",
        "in",
        "    #\"Changed Type\""
    ]

def get_default_page_layout():
    """Get default page layout settings"""
    return {
        "width": 1280,
        "height": 720,
        "displayOption": 0,
        "background": [
            {
                "color": {
                    "expr": {
                        "ThemeDataColor": {
                            "ColorId": 0,
                            "Percent": 0
                        }
                    }
                }
            }
        ]
    }
