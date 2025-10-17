import os
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch and transformers not available. Using fallback mode.")
import git
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any, Optional

class AIModels:
    """Manages AI model loading and inference"""
    
    def __init__(self):
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.device = "cpu"
        self.model_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models only if torch is available
        if TORCH_AVAILABLE:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._load_qwen_model()
            except Exception as e:
                self.logger.error(f"Could not load AI models: {e}")
                self.model_loaded = False
        else:
            self.logger.warning("PyTorch not available, using fallback mode")
    
    def _load_qwen_model(self):
        """Load Qwen model from HuggingFace"""
        try:
            model_name = "Qwen/Qwen2-1.5B-Instruct"  # Using smaller model for better performance
            
            self.logger.info(f"Loading Qwen model: {model_name}")
            
            # Load tokenizer
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Add pad token if it doesn't exist
            if self.qwen_tokenizer.pad_token is None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
            # Load model with reduced precision for memory efficiency
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.qwen_model = self.qwen_model.to(self.device)
            
            self.model_loaded = True
            self.logger.info("Qwen model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Qwen model: {str(e)}")
            self.model_loaded = False
            raise
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text using Qwen model"""
        if not TORCH_AVAILABLE or not self.model_loaded:
            return self._fallback_text_generation(prompt)
        
        try:
            # Prepare the prompt with proper formatting
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            inputs = self.qwen_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.pad_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.qwen_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def analyze_data_with_llm(self, data_summary: Dict[str, Any], sample_data: str) -> Dict[str, Any]:
        """Use LLM to analyze data and provide insights"""
        prompt = f"""
        Analyze the following dataset and provide insights:
        
        Dataset Summary:
        - Rows: {data_summary.get('row_count', 0)}
        - Columns: {data_summary.get('column_count', 0)}
        - Numeric columns: {data_summary.get('numeric_columns', 0)}
        - Text columns: {data_summary.get('text_columns', 0)}
        - Missing data: {data_summary.get('missing_data_percentage', 0):.2f}%
        
        Sample Data:
        {sample_data}
        
        Please provide:
        1. A brief summary of what this dataset contains
        2. Key patterns or insights you notice
        3. Potential data quality issues
        4. Suggested visualizations for this data
        
        Response format: Provide a clear, structured analysis.
        """
        
        response = self.generate_text(prompt, max_length=600)
        
        # Parse the response into structured format
        return {
            'llm_analysis': response,
            'summary': self._extract_summary(response),
            'insights': self._extract_insights(response),
            'quality_issues': self._extract_quality_issues(response),
            'viz_suggestions': self._extract_viz_suggestions(response)
        }
    
    def generate_dashboard_design(self, data_analysis: Dict, design_prompt: str) -> Dict[str, Any]:
        """Generate dashboard design based on data analysis and user prompt"""
        prompt = f"""
        Based on the following data analysis and user requirements, design a Power BI dashboard:
        
        Data Analysis:
        {data_analysis.get('summary', 'No summary available')}
        
        User Requirements:
        {design_prompt}
        
        Please provide:
        1. Dashboard layout structure
        2. Recommended visualizations with specific columns
        3. KPI cards to include
        4. Filter recommendations
        5. Color scheme suggestions
        
        Be specific about which columns to use for each visualization.
        """
        
        response = self.generate_text(prompt, max_length=800)
        
        return {
            'design_response': response,
            'layout': self._extract_layout(response),
            'visualizations': self._extract_visualizations(response),
            'kpis': self._extract_kpis(response),
            'filters': self._extract_filters(response)
        }
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from LLM response"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'summary' in line.lower() or 'contains' in line.lower():
                # Return the next few lines as summary
                summary_lines = lines[i:i+3]
                return ' '.join(summary_lines).strip()
        return "No summary extracted"
    
    def _extract_insights(self, text: str) -> list:
        """Extract insights from LLM response"""
        insights = []
        lines = text.split('\n')
        in_insights_section = False
        
        for line in lines:
            if 'pattern' in line.lower() or 'insight' in line.lower():
                in_insights_section = True
                continue
            if in_insights_section and line.strip():
                if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                    insights.append(line.strip())
                elif not any(char.isdigit() for char in line[:3]):
                    break
        
        return insights[:5]  # Limit to 5 insights
    
    def _extract_quality_issues(self, text: str) -> list:
        """Extract data quality issues from LLM response"""
        issues = []
        lines = text.split('\n')
        in_quality_section = False
        
        for line in lines:
            if 'quality' in line.lower() or 'issue' in line.lower():
                in_quality_section = True
                continue
            if in_quality_section and line.strip():
                if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                    issues.append(line.strip())
        
        return issues[:3]  # Limit to 3 issues
    
    def _extract_viz_suggestions(self, text: str) -> list:
        """Extract visualization suggestions from LLM response"""
        suggestions = []
        lines = text.split('\n')
        in_viz_section = False
        
        for line in lines:
            if 'visualiz' in line.lower() or 'chart' in line.lower() or 'graph' in line.lower():
                in_viz_section = True
                continue
            if in_viz_section and line.strip():
                if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                    suggestions.append(line.strip())
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def _extract_layout(self, text: str) -> Dict[str, str]:
        """Extract layout information from design response"""
        layout = {}
        lines = text.split('\n')
        
        for line in lines:
            if 'layout' in line.lower():
                layout['structure'] = line.strip()
                break
        
        return layout
    
    def _extract_visualizations(self, text: str) -> list:
        """Extract visualization recommendations from design response"""
        visualizations = []
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['chart', 'graph', 'visualization', 'plot']):
                if line.strip() and not line.lower().startswith('please'):
                    visualizations.append(line.strip())
        
        return visualizations[:8]  # Limit to 8 visualizations
    
    def _extract_kpis(self, text: str) -> list:
        """Extract KPI recommendations from design response"""
        kpis = []
        lines = text.split('\n')
        
        for line in lines:
            if 'kpi' in line.lower() or 'metric' in line.lower() or 'card' in line.lower():
                if line.strip() and not line.lower().startswith('please'):
                    kpis.append(line.strip())
        
        return kpis[:4]  # Limit to 4 KPIs
    
    def _extract_filters(self, text: str) -> list:
        """Extract filter recommendations from design response"""
        filters = []
        lines = text.split('\n')
        
        for line in lines:
            if 'filter' in line.lower() or 'slicer' in line.lower():
                if line.strip() and not line.lower().startswith('please'):
                    filters.append(line.strip())
        
        return filters[:5]  # Limit to 5 filters
    
    def _fallback_text_generation(self, prompt: str) -> str:
        """Fallback text generation when torch is not available"""
        # Simple rule-based responses
        if 'dataset' in prompt.lower() or 'data' in prompt.lower():
            return """This dataset appears to contain structured data with multiple columns. 
            Key insights: The data seems suitable for various visualizations including bar charts, line charts, and tables.
            Recommended visualizations: Consider using bar charts for categorical comparisons, line charts for trends over time, and tables for detailed data views."""
        
        if 'dashboard' in prompt.lower() or 'design' in prompt.lower():
            return """Dashboard Design Recommendations:
            1. Layout: Use a grid-based layout with KPI cards at the top
            2. Visualizations: Include bar charts for comparisons, line charts for trends, and pie charts for proportions
            3. KPIs: Display key metrics like totals, averages, and counts
            4. Filters: Add interactive slicers for date ranges and categories
            5. Color scheme: Use a professional blue color scheme for consistency"""
        
        return "AI model not available. Using basic recommendations based on data structure."
