import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re
from typing import Dict, List, Any

class LLMAgent:
    """
    Agent that uses Qwen LLM for natural language understanding and processing
    """
    
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Qwen model from HuggingFace"""
        try:
            print(f"Loading Qwen model: {self.model_name}")
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("Qwen model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            print("Falling back to rule-based NLP processing")
            self.model = None
            self.pipeline = None
    
    def parse_requirements(self, prompt: str, available_columns: List[str], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Parse user requirements from natural language prompt
        """
        if self.pipeline:
            return self._parse_with_llm(prompt, available_columns, temperature)
        else:
            return self._parse_with_rules(prompt, available_columns)
    
    def _parse_with_llm(self, prompt: str, available_columns: List[str], temperature: float) -> Dict[str, Any]:
        """Use Qwen LLM for requirement parsing"""
        try:
            # Construct system prompt for dashboard analysis
            system_prompt = f"""You are an expert Power BI dashboard designer. Analyze the user's request and extract structured requirements.

Available data columns: {', '.join(available_columns)}

Parse the following request and return a JSON response with this structure:
{{
    "objective": "Main goal of the dashboard",
    "visualizations": [
        {{
            "type": "chart_type",
            "description": "what this visualization shows",
            "fields": ["column1", "column2"],
            "aggregation": "sum/avg/count/etc",
            "priority": 1-5
        }}
    ],
    "kpis": [
        {{
            "name": "KPI name",
            "calculation": "how to calculate",
            "format": "currency/percentage/number"
        }}
    ],
    "filters": ["suggested filter fields"],
    "layout_preference": "executive/detailed/operational",
    "color_scheme": "professional/vibrant/corporate"
}}

User Request: {prompt}

Response (JSON only):"""

            # Generate response
            response = self.pipeline(
                system_prompt,
                max_new_tokens=1000,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = generated_text[json_start:json_end]
                parsed_requirements = json.loads(json_str)
                
                # Validate and clean requirements
                return self._validate_requirements(parsed_requirements, available_columns)
            else:
                print("Could not extract JSON from LLM response, falling back to rules")
                return self._parse_with_rules(prompt, available_columns)
                
        except Exception as e:
            print(f"Error in LLM parsing: {e}")
            return self._parse_with_rules(prompt, available_columns)
    
    def _parse_with_rules(self, prompt: str, available_columns: List[str]) -> Dict[str, Any]:
        """Fallback rule-based requirement parsing"""
        prompt_lower = prompt.lower()
        
        # Extract visualization types mentioned
        viz_keywords = {
            'bar': ['bar', 'column', 'histogram'],
            'line': ['line', 'trend', 'time series', 'temporal'],
            'pie': ['pie', 'donut', 'proportion'],
            'scatter': ['scatter', 'correlation', 'relationship'],
            'table': ['table', 'grid', 'list'],
            'card': ['kpi', 'metric', 'total', 'sum', 'count', 'average'],
            'map': ['map', 'geographic', 'location', 'region']
        }
        
        suggested_visualizations = []
        
        for viz_type, keywords in viz_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                # Find relevant columns for this visualization
                relevant_cols = self._find_relevant_columns(viz_type, available_columns, prompt_lower)
                if relevant_cols:
                    suggested_visualizations.append({
                        'type': viz_type,
                        'description': f"{viz_type.title()} chart showing {', '.join(relevant_cols[:2])}",
                        'fields': relevant_cols[:3],
                        'aggregation': self._suggest_aggregation(viz_type, relevant_cols),
                        'priority': self._calculate_priority(viz_type, prompt_lower)
                    })
        
        # Extract KPIs
        kpi_keywords = ['total', 'sum', 'average', 'count', 'maximum', 'minimum', 'percentage']
        kpis = []
        
        for keyword in kpi_keywords:
            if keyword in prompt_lower:
                relevant_cols = [col for col in available_columns 
                               if any(term in col.lower() for term in ['sales', 'revenue', 'amount', 'price', 'cost', 'value'])]
                if relevant_cols:
                    kpis.append({
                        'name': f"{keyword.title()} {relevant_cols[0]}",
                        'calculation': keyword,
                        'format': self._suggest_format(relevant_cols[0])
                    })
        
        # Extract filters
        suggested_filters = [col for col in available_columns 
                           if any(term in col.lower() for term in ['category', 'type', 'region', 'department', 'status'])]
        
        return {
            'objective': f"Dashboard based on user requirements: {prompt[:100]}...",
            'visualizations': suggested_visualizations,
            'kpis': kpis,
            'filters': suggested_filters[:5],
            'layout_preference': self._suggest_layout(prompt_lower),
            'color_scheme': self._suggest_color_scheme(prompt_lower)
        }
    
    def _find_relevant_columns(self, viz_type: str, available_columns: List[str], prompt: str) -> List[str]:
        """Find columns relevant to a specific visualization type"""
        numeric_indicators = ['amount', 'price', 'cost', 'value', 'sales', 'revenue', 'quantity', 'count', 'total']
        categorical_indicators = ['category', 'type', 'name', 'status', 'region', 'department']
        date_indicators = ['date', 'time', 'year', 'month', 'day']
        
        relevant_cols = []
        
        if viz_type in ['bar', 'line', 'scatter']:
            # Need both categorical and numeric
            numeric_cols = [col for col in available_columns 
                          if any(indicator in col.lower() for indicator in numeric_indicators)]
            categorical_cols = [col for col in available_columns 
                              if any(indicator in col.lower() for indicator in categorical_indicators)]
            
            relevant_cols.extend(categorical_cols[:2])
            relevant_cols.extend(numeric_cols[:2])
            
        elif viz_type == 'pie':
            # Need categorical for segments and numeric for values
            categorical_cols = [col for col in available_columns 
                              if any(indicator in col.lower() for indicator in categorical_indicators)]
            numeric_cols = [col for col in available_columns 
                          if any(indicator in col.lower() for indicator in numeric_indicators)]
            
            relevant_cols.extend(categorical_cols[:1])
            relevant_cols.extend(numeric_cols[:1])
            
        elif viz_type == 'card':
            # Need numeric columns for KPIs
            numeric_cols = [col for col in available_columns 
                          if any(indicator in col.lower() for indicator in numeric_indicators)]
            relevant_cols.extend(numeric_cols[:1])
            
        elif viz_type == 'table':
            # Can use any columns, prioritize those mentioned in prompt
            mentioned_cols = [col for col in available_columns if col.lower() in prompt]
            relevant_cols.extend(mentioned_cols[:5])
            if len(relevant_cols) < 3:
                relevant_cols.extend([col for col in available_columns if col not in relevant_cols][:3-len(relevant_cols)])
        
        return relevant_cols
    
    def _suggest_aggregation(self, viz_type: str, columns: List[str]) -> str:
        """Suggest appropriate aggregation for visualization"""
        if not columns:
            return 'none'
        
        first_col = columns[0].lower()
        
        if 'count' in first_col or 'quantity' in first_col:
            return 'sum'
        elif 'amount' in first_col or 'revenue' in first_col or 'sales' in first_col:
            return 'sum'
        elif 'price' in first_col or 'cost' in first_col:
            return 'average'
        elif viz_type == 'card':
            return 'sum'
        else:
            return 'sum'
    
    def _calculate_priority(self, viz_type: str, prompt: str) -> int:
        """Calculate priority based on mention in prompt"""
        viz_mentions = prompt.count(viz_type)
        
        if viz_mentions > 1:
            return 1  # High priority
        elif viz_mentions == 1:
            return 2  # Medium priority
        elif viz_type in ['card', 'table']:
            return 3  # Default medium for common types
        else:
            return 4  # Lower priority
    
    def _suggest_format(self, column_name: str) -> str:
        """Suggest format based on column name"""
        column_lower = column_name.lower()
        
        if any(term in column_lower for term in ['revenue', 'sales', 'amount', 'price', 'cost']):
            return 'currency'
        elif any(term in column_lower for term in ['rate', 'percentage', 'percent']):
            return 'percentage'
        else:
            return 'number'
    
    def _suggest_layout(self, prompt: str) -> str:
        """Suggest dashboard layout based on prompt"""
        if any(term in prompt for term in ['executive', 'summary', 'overview', 'high-level']):
            return 'executive'
        elif any(term in prompt for term in ['detailed', 'analysis', 'deep', 'comprehensive']):
            return 'detailed'
        elif any(term in prompt for term in ['operational', 'daily', 'monitoring', 'real-time']):
            return 'operational'
        else:
            return 'detailed'
    
    def _suggest_color_scheme(self, prompt: str) -> str:
        """Suggest color scheme based on prompt"""
        if any(term in prompt for term in ['corporate', 'business', 'professional']):
            return 'corporate'
        elif any(term in prompt for term in ['vibrant', 'colorful', 'bright']):
            return 'vibrant'
        else:
            return 'professional'
    
    def _validate_requirements(self, requirements: Dict[str, Any], available_columns: List[str]) -> Dict[str, Any]:
        """Validate and clean parsed requirements"""
        # Ensure all required fields exist
        if 'visualizations' not in requirements:
            requirements['visualizations'] = []
        
        if 'kpis' not in requirements:
            requirements['kpis'] = []
        
        # Validate column references
        for viz in requirements['visualizations']:
            if 'fields' in viz:
                viz['fields'] = [field for field in viz['fields'] if field in available_columns]
        
        # Remove empty visualizations
        requirements['visualizations'] = [viz for viz in requirements['visualizations'] 
                                        if viz.get('fields')]
        
        return requirements
