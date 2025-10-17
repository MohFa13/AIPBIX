import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class RecursiveAgent:
    """
    Agent that uses TinyRecursiveModels for multi-step reasoning about dashboard creation
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_context_length = 2048
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the recursive reasoning model"""
        try:
            # Try to use TinyRecursiveModels if available
            if os.path.exists("TinyRecursiveModels"):
                sys.path.append("TinyRecursiveModels")
                # Import TinyRecursiveModels components if available
                try:
                    from models import TinyRecursiveModel
                    self.model = TinyRecursiveModel()
                    print("TinyRecursiveModels loaded successfully")
                except ImportError:
                    print("TinyRecursiveModels not available, using fallback reasoning")
                    self.model = None
            else:
                self.model = None
                
        except Exception as e:
            print(f"Error initializing recursive model: {e}")
            self.model = None
    
    def reason_about_dashboard(self, requirements, data, max_steps=5):
        """
        Perform multi-step reasoning about dashboard creation
        """
        reasoning_steps = []
        
        # Step 1: Analyze data structure
        step1 = self._analyze_data_structure(data)
        reasoning_steps.append(f"Data Structure Analysis: {step1}")
        
        # Step 2: Map requirements to data capabilities
        step2 = self._map_requirements_to_data(requirements, data)
        reasoning_steps.append(f"Requirements Mapping: {step2}")
        
        # Step 3: Identify visualization opportunities
        step3 = self._identify_visualization_opportunities(requirements, data)
        reasoning_steps.append(f"Visualization Opportunities: {step3}")
        
        # Step 4: Plan dashboard layout
        step4 = self._plan_dashboard_layout(requirements, data)
        reasoning_steps.append(f"Dashboard Layout Planning: {step4}")
        
        # Step 5: Validate feasibility
        step5 = self._validate_feasibility(requirements, data)
        reasoning_steps.append(f"Feasibility Validation: {step5}")
        
        # If TinyRecursiveModels is available, use it for additional reasoning
        if self.model and hasattr(self.model, 'recursive_reasoning'):
            try:
                enhanced_steps = self._enhance_with_recursive_model(reasoning_steps, requirements, data)
                reasoning_steps.extend(enhanced_steps)
            except Exception as e:
                print(f"Error in recursive model enhancement: {e}")
        
        return {
            'steps': reasoning_steps,
            'total_steps': len(reasoning_steps),
            'recommendations': self._generate_recommendations(reasoning_steps)
        }
    
    def _analyze_data_structure(self, data):
        """Analyze the structure and characteristics of the data"""
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'data_density': (data.notna().sum().sum() / (len(data) * len(data.columns)))
        }
        
        return f"Dataset has {analysis['total_rows']} rows and {analysis['total_columns']} columns. " \
               f"Contains {analysis['numeric_columns']} numeric, {analysis['categorical_columns']} categorical, " \
               f"and {analysis['datetime_columns']} datetime fields. Data density: {analysis['data_density']:.2%}"
    
    def _map_requirements_to_data(self, requirements, data):
        """Map user requirements to available data capabilities"""
        available_fields = set(data.columns.str.lower())
        
        mappings = []
        for viz in requirements.get('visualizations', []):
            required_fields = [field.lower() for field in viz.get('fields', [])]
            available = all(field in available_fields for field in required_fields)
            mappings.append({
                'visualization': viz.get('type', 'Unknown'),
                'feasible': available,
                'missing_fields': [f for f in required_fields if f not in available_fields]
            })
        
        feasible_count = sum(1 for m in mappings if m['feasible'])
        return f"{feasible_count}/{len(mappings)} requested visualizations are feasible with current data"
    
    def _identify_visualization_opportunities(self, requirements, data):
        """Identify potential visualization opportunities based on data"""
        opportunities = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Time series opportunities
        if datetime_cols and numeric_cols:
            opportunities.append("Time series analysis with line charts")
        
        # Categorical analysis opportunities
        if categorical_cols and numeric_cols:
            opportunities.append("Category comparison with bar/column charts")
        
        # Distribution opportunities
        if len(numeric_cols) >= 1:
            opportunities.append("Distribution analysis with histograms")
        
        # Correlation opportunities
        if len(numeric_cols) >= 2:
            opportunities.append("Correlation analysis with scatter plots")
        
        # Summary opportunities
        if numeric_cols:
            opportunities.append("KPI cards for key metrics")
        
        return f"Identified {len(opportunities)} visualization opportunities: {', '.join(opportunities)}"
    
    def _plan_dashboard_layout(self, requirements, data):
        """Plan the optimal dashboard layout"""
        viz_count = len(requirements.get('visualizations', []))
        
        if viz_count <= 4:
            layout = "2x2 grid layout for optimal viewing"
        elif viz_count <= 6:
            layout = "3x2 grid layout with hierarchical importance"
        elif viz_count <= 9:
            layout = "3x3 grid layout with categorized sections"
        else:
            layout = "Multi-page layout with themed pages"
        
        return f"Recommended layout: {layout} for {viz_count} visualizations"
    
    def _validate_feasibility(self, requirements, data):
        """Validate the feasibility of the dashboard requirements"""
        issues = []
        
        # Check data volume
        if len(data) > 1000000:
            issues.append("Large dataset may require performance optimization")
        
        # Check missing data
        missing_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_percentage > 0.1:
            issues.append(f"High missing data percentage ({missing_percentage:.1%})")
        
        # Check visualization complexity
        viz_count = len(requirements.get('visualizations', []))
        if viz_count > 15:
            issues.append("High visualization count may impact performance")
        
        if issues:
            return f"Potential issues identified: {'; '.join(issues)}"
        else:
            return "All requirements appear feasible with current data"
    
    def _enhance_with_recursive_model(self, reasoning_steps, requirements, data):
        """Use TinyRecursiveModels for enhanced reasoning if available"""
        enhanced_steps = []
        
        try:
            # Prepare context for recursive reasoning
            context = {
                'data_summary': f"Dataset with {len(data)} rows and {len(data.columns)} columns",
                'requirements': str(requirements),
                'initial_reasoning': reasoning_steps
            }
            
            # Apply recursive reasoning if model is available
            if hasattr(self.model, 'reason'):
                result = self.model.reason(context, max_depth=3)
                enhanced_steps.append(f"Recursive Model Enhancement: {result}")
            else:
                # Fallback to rule-based enhancement
                enhanced_steps.append("Applied heuristic reasoning for optimization recommendations")
                
        except Exception as e:
            enhanced_steps.append(f"Recursive reasoning attempted but encountered: {str(e)}")
        
        return enhanced_steps
    
    def _generate_recommendations(self, reasoning_steps):
        """Generate actionable recommendations based on reasoning"""
        recommendations = []
        
        # Extract key insights from reasoning steps
        for step in reasoning_steps:
            if "feasible" in step.lower():
                recommendations.append("Focus on feasible visualizations first, then explore alternatives")
            if "performance" in step.lower():
                recommendations.append("Consider data aggregation and filtering for better performance")
            if "missing data" in step.lower():
                recommendations.append("Implement data quality measures and highlight data gaps")
            if "opportunities" in step.lower():
                recommendations.append("Leverage identified visualization opportunities for deeper insights")
        
        # Add general recommendations
        recommendations.extend([
            "Include interactive filters for user exploration",
            "Implement consistent color scheme across visualizations",
            "Add tooltips and drill-through capabilities where appropriate",
            "Consider mobile-friendly responsive design"
        ])
        
        return list(set(recommendations))  # Remove duplicates
