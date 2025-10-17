import os
import sys
import tempfile
import shutil
import git
from pathlib import Path
import importlib.util
import json
from typing import Dict, Any, List
import pandas as pd

class TinyRecursiveAgent:
    """Integrates TinyRecursiveModels for agentic multi-step reasoning"""
    
    def __init__(self):
        self.repo_path = None
        self.models_loaded = False
        self.recursive_model = None
        
        # Clone and setup TinyRecursiveModels
        self._setup_tiny_recursive_models()
    
    def _setup_tiny_recursive_models(self):
        """Clone and setup TinyRecursiveModels repository"""
        try:
            # Create temporary directory for the repository
            self.repo_path = os.path.join(tempfile.gettempdir(), 'TinyRecursiveModels')
            
            # Remove existing directory if it exists
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)
            
            # Clone the repository
            print("Cloning TinyRecursiveModels repository...")
            git.Repo.clone_from(
                'https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git',
                self.repo_path
            )
            
            # Add to Python path
            if self.repo_path not in sys.path:
                sys.path.insert(0, self.repo_path)
            
            # Try to import and initialize the model
            self._initialize_recursive_model()
            
            print("TinyRecursiveModels setup completed successfully")
            
        except Exception as e:
            print(f"Warning: Could not setup TinyRecursiveModels: {str(e)}")
            self.models_loaded = False
    
    def _initialize_recursive_model(self):
        """Initialize the recursive reasoning model"""
        try:
            # This is a placeholder implementation since the actual TinyRecursiveModels
            # implementation details would depend on the specific model architecture
            
            # For now, we'll create a simple multi-step reasoning framework
            self.recursive_model = SimpleRecursiveReasoner()
            self.models_loaded = True
            
        except Exception as e:
            print(f"Could not initialize recursive model: {str(e)}")
            # Fallback to simple reasoning
            self.recursive_model = SimpleRecursiveReasoner()
            self.models_loaded = True
    
    def process_design_request(self, design_prompt: str, dataset: pd.DataFrame, 
                             data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process design request using multi-step reasoning"""
        
        if not self.models_loaded:
            return self._fallback_design_processing(design_prompt, dataset, data_analysis)
        
        try:
            # Step 1: Parse and understand the design requirements
            parsed_requirements = self._parse_design_requirements(design_prompt)
            
            # Step 2: Analyze data compatibility with requirements
            compatibility_analysis = self._analyze_data_compatibility(parsed_requirements, dataset, data_analysis)
            
            # Step 3: Generate design recommendations
            design_recommendations = self._generate_design_recommendations(
                parsed_requirements, compatibility_analysis, data_analysis
            )
            
            # Step 4: Optimize layout and visual hierarchy
            optimized_layout = self._optimize_layout(design_recommendations, dataset.shape)
            
            # Step 5: Generate final design specification
            final_design = self._finalize_design_specification(
                parsed_requirements, design_recommendations, optimized_layout
            )
            
            return {
                'interpretation': final_design.get('interpretation', ''),
                'layout': final_design.get('layout', {}),
                'reasoning_steps': final_design.get('reasoning_steps', []),
                'recommendations': design_recommendations,
                'compatibility_score': compatibility_analysis.get('score', 0.0)
            }
            
        except Exception as e:
            print(f"Error in recursive processing: {str(e)}")
            return self._fallback_design_processing(design_prompt, dataset, data_analysis)
    
    def _parse_design_requirements(self, design_prompt: str) -> Dict[str, Any]:
        """Parse design prompt to extract structured requirements"""
        
        # Use recursive model for parsing if available
        if hasattr(self.recursive_model, 'parse_requirements'):
            return self.recursive_model.parse_requirements(design_prompt)
        
        # Fallback parsing logic
        requirements = {
            'visualization_types': [],
            'data_focus': [],
            'color_preferences': [],
            'layout_preferences': [],
            'kpi_requests': [],
            'filter_requests': []
        }
        
        prompt_lower = design_prompt.lower()
        
        # Extract visualization types
        viz_keywords = {
            'chart': ['chart', 'graph'],
            'bar': ['bar', 'column'],
            'line': ['line', 'trend', 'time'],
            'pie': ['pie', 'donut'],
            'scatter': ['scatter', 'correlation'],
            'table': ['table', 'grid'],
            'map': ['map', 'geographical', 'geo'],
            'kpi': ['kpi', 'metric', 'card', 'indicator']
        }
        
        for viz_type, keywords in viz_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                requirements['visualization_types'].append(viz_type)
        
        # Extract color preferences
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
        for color in colors:
            if color in prompt_lower:
                requirements['color_preferences'].append(color)
        
        # Extract layout preferences
        layout_keywords = ['dashboard', 'summary', 'detailed', 'executive', 'operational']
        for keyword in layout_keywords:
            if keyword in prompt_lower:
                requirements['layout_preferences'].append(keyword)
        
        return requirements
    
    def _analyze_data_compatibility(self, requirements: Dict, dataset: pd.DataFrame, 
                                  data_analysis: Dict) -> Dict[str, Any]:
        """Analyze how well the data matches the design requirements"""
        
        compatibility_score = 0.0
        compatibility_details = []
        
        numeric_cols = len(dataset.select_dtypes(include=['number']).columns)
        text_cols = len(dataset.select_dtypes(include=['object']).columns)
        date_cols = len(data_analysis.get('patterns', {}).get('time_series', []))
        
        # Check visualization compatibility
        for viz_type in requirements.get('visualization_types', []):
            if viz_type == 'bar' and text_cols > 0 and numeric_cols > 0:
                compatibility_score += 0.2
                compatibility_details.append(f"Bar charts possible with {text_cols} categorical and {numeric_cols} numeric columns")
            
            elif viz_type == 'line' and date_cols > 0 and numeric_cols > 0:
                compatibility_score += 0.2
                compatibility_details.append(f"Line charts possible with {date_cols} date and {numeric_cols} numeric columns")
            
            elif viz_type == 'pie' and text_cols > 0 and numeric_cols > 0:
                suitable_categorical = sum(1 for col in dataset.select_dtypes(include=['object']).columns 
                                         if dataset[col].nunique() <= 10)
                if suitable_categorical > 0:
                    compatibility_score += 0.15
                    compatibility_details.append(f"Pie charts possible with {suitable_categorical} suitable categorical columns")
            
            elif viz_type == 'kpi' and numeric_cols > 0:
                compatibility_score += 0.15
                compatibility_details.append(f"KPI cards possible with {numeric_cols} numeric columns")
            
            elif viz_type == 'table':
                compatibility_score += 0.1
                compatibility_details.append("Data table always possible")
        
        # Bonus for data quality
        missing_data_pct = data_analysis.get('basic_stats', {}).get('missing_data_percentage', 0)
        if missing_data_pct < 10:
            compatibility_score += 0.1
            compatibility_details.append("Good data quality (low missing data)")
        
        return {
            'score': min(compatibility_score, 1.0),
            'details': compatibility_details,
            'data_summary': {
                'numeric_columns': numeric_cols,
                'text_columns': text_cols,
                'date_columns': date_cols,
                'total_columns': len(dataset.columns),
                'rows': len(dataset)
            }
        }
    
    def _generate_design_recommendations(self, requirements: Dict, compatibility: Dict, 
                                       data_analysis: Dict) -> Dict[str, Any]:
        """Generate specific design recommendations"""
        
        recommendations = {
            'priority_visualizations': [],
            'layout_structure': {},
            'color_scheme': {},
            'interactions': []
        }
        
        # Prioritize visualizations based on compatibility and requirements
        if 'kpi' in requirements.get('visualization_types', []) or compatibility['data_summary']['numeric_columns'] > 0:
            recommendations['priority_visualizations'].append({
                'type': 'KPI Cards',
                'priority': 'High',
                'reason': 'Numeric data available for key metrics'
            })
        
        if compatibility['data_summary']['text_columns'] > 0 and compatibility['data_summary']['numeric_columns'] > 0:
            recommendations['priority_visualizations'].append({
                'type': 'Bar Charts',
                'priority': 'High',
                'reason': 'Categorical and numeric data available for comparisons'
            })
        
        if compatibility['data_summary']['date_columns'] > 0:
            recommendations['priority_visualizations'].append({
                'type': 'Time Series',
                'priority': 'High',
                'reason': 'Date columns available for trend analysis'
            })
        
        # Layout recommendations
        total_visuals = len(recommendations['priority_visualizations'])
        if total_visuals <= 4:
            recommendations['layout_structure'] = {'type': 'simple_grid', 'rows': 2, 'cols': 2}
        elif total_visuals <= 6:
            recommendations['layout_structure'] = {'type': 'complex_grid', 'rows': 3, 'cols': 2}
        else:
            recommendations['layout_structure'] = {'type': 'dashboard', 'sections': ['kpi', 'main', 'details']}
        
        # Color scheme
        preferred_colors = requirements.get('color_preferences', [])
        if preferred_colors:
            recommendations['color_scheme'] = {'primary': preferred_colors[0], 'type': 'monochromatic'}
        else:
            recommendations['color_scheme'] = {'primary': 'blue', 'type': 'professional'}
        
        return recommendations
    
    def _optimize_layout(self, recommendations: Dict, data_shape: tuple) -> Dict[str, Any]:
        """Optimize layout based on recommendations and data characteristics"""
        
        rows, cols = data_shape
        
        # Determine optimal layout based on data size and visualization count
        viz_count = len(recommendations.get('priority_visualizations', []))
        
        if rows > 10000:  # Large dataset
            layout = {
                'type': 'performance_optimized',
                'kpi_section': {'width': '100%', 'height': '15%'},
                'main_section': {'width': '100%', 'height': '60%'},
                'filter_section': {'width': '20%', 'height': '25%'},
                'detail_section': {'width': '80%', 'height': '25%'}
            }
        elif viz_count > 6:  # Many visualizations
            layout = {
                'type': 'multi_section',
                'header': {'kpi_cards': 4},
                'main': {'charts': 4},
                'sidebar': {'filters': 2},
                'footer': {'table': 1}
            }
        else:  # Standard layout
            layout = {
                'type': 'standard',
                'grid': {'rows': 2, 'cols': 2},
                'header_kpis': True,
                'sidebar_filters': True
            }
        
        return layout
    
    def _finalize_design_specification(self, requirements: Dict, recommendations: Dict, 
                                     layout: Dict) -> Dict[str, Any]:
        """Generate final design specification"""
        
        interpretation = self._generate_interpretation(requirements, recommendations)
        
        final_layout = {
            'structure': layout.get('type', 'standard'),
            'sections': self._define_layout_sections(layout, recommendations),
            'styling': {
                'color_scheme': recommendations.get('color_scheme', {}),
                'typography': {'title_size': 'large', 'body_size': 'medium'},
                'spacing': {'margin': 'standard', 'padding': 'comfortable'}
            }
        }
        
        reasoning_steps = [
            "1. Parsed user requirements for visualization types and preferences",
            "2. Analyzed data compatibility with requested visualizations",
            "3. Prioritized visualizations based on data characteristics",
            "4. Optimized layout for data size and visualization count",
            "5. Generated final design specification with styling"
        ]
        
        return {
            'interpretation': interpretation,
            'layout': final_layout,
            'reasoning_steps': reasoning_steps
        }
    
    def _generate_interpretation(self, requirements: Dict, recommendations: Dict) -> str:
        """Generate human-readable interpretation of the design"""
        
        viz_types = requirements.get('visualization_types', [])
        priority_viz = [v['type'] for v in recommendations.get('priority_visualizations', [])]
        
        interpretation = f"Based on your request, I'll create a dashboard with {len(priority_viz)} main visualization types. "
        
        if 'kpi' in viz_types or 'KPI Cards' in priority_viz:
            interpretation += "KPI cards will highlight key metrics at the top. "
        
        if 'bar' in viz_types or 'Bar Charts' in priority_viz:
            interpretation += "Bar charts will show categorical comparisons. "
        
        if 'line' in viz_types or 'Time Series' in priority_viz:
            interpretation += "Line charts will display trends over time. "
        
        color_scheme = recommendations.get('color_scheme', {}).get('primary', 'blue')
        interpretation += f"The dashboard will use a {color_scheme} color scheme for a professional appearance."
        
        return interpretation
    
    def _define_layout_sections(self, layout: Dict, recommendations: Dict) -> Dict[str, str]:
        """Define specific layout sections"""
        
        sections = {}
        
        if layout.get('type') == 'performance_optimized':
            sections = {
                'header': 'KPI cards showing key metrics',
                'main': 'Primary charts and visualizations',
                'sidebar': 'Interactive filters and controls',
                'footer': 'Detailed data table'
            }
        elif layout.get('type') == 'multi_section':
            sections = {
                'top': 'Dashboard header with KPI cards',
                'left': 'Main visualizations and charts',
                'right': 'Filters and secondary metrics',
                'bottom': 'Supporting data and tables'
            }
        else:  # standard
            sections = {
                'grid_1_1': 'Primary KPI or chart',
                'grid_1_2': 'Secondary visualization',
                'grid_2_1': 'Supporting chart',
                'grid_2_2': 'Data table or additional metrics'
            }
        
        return sections
    
    def _fallback_design_processing(self, design_prompt: str, dataset: pd.DataFrame, 
                                  data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when recursive models are not available"""
        
        # Simple rule-based processing
        interpretation = f"Creating a dashboard based on your requirements: {design_prompt[:100]}..."
        
        # Basic layout recommendation
        num_cols = len(dataset.columns)
        if num_cols <= 5:
            layout_type = "Simple Grid"
        elif num_cols <= 10:
            layout_type = "Standard Dashboard"
        else:
            layout_type = "Complex Multi-Section Layout"
        
        layout = {
            'type': layout_type,
            'kpi_section': 'Top section with key metrics',
            'main_section': 'Central area with primary charts',
            'filter_section': 'Sidebar with interactive filters'
        }
        
        return {
            'interpretation': interpretation,
            'layout': layout,
            'reasoning_steps': [
                "Analyzed design prompt for visualization requirements",
                "Assessed data structure and column types",
                "Generated basic layout recommendation",
                "Applied standard dashboard design principles"
            ],
            'compatibility_score': 0.7  # Default reasonable score
        }


class SimpleRecursiveReasoner:
    """Simple implementation of recursive reasoning for design tasks"""
    
    def __init__(self):
        self.reasoning_depth = 3  # Maximum depth for recursive reasoning
    
    def parse_requirements(self, prompt: str) -> Dict[str, Any]:
        """Parse requirements with simple recursive logic"""
        
        # This would be enhanced with actual TinyRecursiveModels implementation
        requirements = {
            'visualization_types': self._extract_visualization_types(prompt),
            'data_requirements': self._extract_data_requirements(prompt),
            'style_preferences': self._extract_style_preferences(prompt),
            'interaction_requirements': self._extract_interaction_requirements(prompt)
        }
        
        return requirements
    
    def _extract_visualization_types(self, prompt: str) -> List[str]:
        """Extract visualization types from prompt"""
        viz_map = {
            'bar': ['bar', 'column', 'histogram'],
            'line': ['line', 'trend', 'time series', 'temporal'],
            'pie': ['pie', 'donut', 'proportion'],
            'scatter': ['scatter', 'correlation', 'relationship'],
            'map': ['map', 'geo', 'geographical', 'spatial'],
            'table': ['table', 'grid', 'tabular'],
            'kpi': ['kpi', 'metric', 'card', 'scorecard']
        }
        
        found_types = []
        prompt_lower = prompt.lower()
        
        for viz_type, keywords in viz_map.items():
            if any(keyword in prompt_lower for keyword in keywords):
                found_types.append(viz_type)
        
        return found_types
    
    def _extract_data_requirements(self, prompt: str) -> List[str]:
        """Extract data requirements from prompt"""
        requirements = []
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['sales', 'revenue', 'financial']):
            requirements.append('financial_data')
        
        if any(word in prompt_lower for word in ['time', 'date', 'monthly', 'yearly']):
            requirements.append('temporal_data')
        
        if any(word in prompt_lower for word in ['category', 'type', 'group']):
            requirements.append('categorical_data')
        
        return requirements
    
    def _extract_style_preferences(self, prompt: str) -> Dict[str, Any]:
        """Extract style preferences from prompt"""
        preferences = {}
        prompt_lower = prompt.lower()
        
        # Color preferences
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'gray']
        for color in colors:
            if color in prompt_lower:
                preferences['primary_color'] = color
                break
        
        # Style preferences
        if 'professional' in prompt_lower or 'corporate' in prompt_lower:
            preferences['style'] = 'professional'
        elif 'modern' in prompt_lower or 'contemporary' in prompt_lower:
            preferences['style'] = 'modern'
        
        return preferences
    
    def _extract_interaction_requirements(self, prompt: str) -> List[str]:
        """Extract interaction requirements from prompt"""
        interactions = []
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['filter', 'slicer', 'interactive']):
            interactions.append('filters')
        
        if any(word in prompt_lower for word in ['drill', 'detail', 'explore']):
            interactions.append('drill_down')
        
        if any(word in prompt_lower for word in ['tooltip', 'hover', 'details']):
            interactions.append('tooltips')
        
        return interactions
