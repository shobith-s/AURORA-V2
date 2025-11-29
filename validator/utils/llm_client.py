"""
LLM Client with support for Hugging Face (FREE alternative to Gemini)
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM validation - supports Ollama, Gemini, and Hugging Face"""
    
    def __init__(
        self, 
        mode: Literal["local", "gemini", "huggingface"] = "huggingface",
        model: str = "qwen2.5:7b",
        api_key: Optional[str] = None,
        api_url: str = "http://localhost:11434/api/generate"
    ):
        self.mode = mode
        self.model = model
        self.api_url = api_url
        self.timeout = 60
        
        if mode == "gemini":
            if not api_key:
                raise ValueError("Gemini API key required for cloud mode")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ Using Gemini 1.5 Flash (cloud)")
        
        elif mode == "huggingface":
            if not api_key:
                raise ValueError("Hugging Face token required")
            from huggingface_hub import InferenceClient
            self.hf_client = InferenceClient(token=api_key)
            self.hf_model = "meta-llama/Llama-3.1-8B-Instruct"
            logger.info(f"✅ Using {self.hf_model} (Hugging Face - FREE)")
        
        else:
            logger.info(f"✅ Using {model} (local)")
        
    def validate_decision(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing decision using LLM"""
        
        if self.mode == "gemini":
            return self._validate_with_gemini(column_info, symbolic_decision)
        elif self.mode == "huggingface":
            return self._validate_with_huggingface(column_info, symbolic_decision)
        else:
            return self._validate_with_ollama(column_info, symbolic_decision)
    
    def _validate_with_huggingface(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using Hugging Face Inference API (FREE)"""
        
        prompt = self._create_validation_prompt(column_info, symbolic_decision)
        
        # Add instruction for JSON output
        prompt += "\n\nRespond ONLY with valid JSON (no markdown, no code blocks):"
        
        try:
            response = self.hf_client.text_generation(
                prompt,
                model=self.hf_model,
                max_new_tokens=500,
                temperature=0.1,
                return_full_text=False
            )
            
            # Parse JSON from response
            # Sometimes models add extra text, try to extract JSON
            response_text = response.strip()
            
            # Try to find JSON in response
            if '{' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
            else:
                # Fallback: trust symbolic
                return self._fallback_response(symbolic_decision)
            
            return {
                'is_correct': result.get('is_correct', True),
                'correct_action': result.get('correct_action', symbolic_decision['action']),
                'reasoning': result.get('reasoning', ''),
                'llm_confidence': result.get('confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Hugging Face validation failed: {e}")
            return self._fallback_response(symbolic_decision)
    
    def _validate_with_gemini(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using Google Gemini API"""
        
        prompt = self._create_validation_prompt(column_info, symbolic_decision)
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 500
                }
            )
            
            # Parse JSON from response
            result = json.loads(response.text)
            return {
                'is_correct': result.get('is_correct', True),
                'correct_action': result.get('correct_action', symbolic_decision['action']),
                'reasoning': result.get('reasoning', ''),
                'llm_confidence': result.get('confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            return self._fallback_response(symbolic_decision)
    
    def _validate_with_ollama(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using local Ollama"""
        
        prompt = self._create_validation_prompt(column_info, symbolic_decision)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = json.loads(response.json()['response'])
                return {
                    'is_correct': result.get('is_correct', True),
                    'correct_action': result.get('correct_action', symbolic_decision['action']),
                    'reasoning': result.get('reasoning', ''),
                    'llm_confidence': result.get('confidence', 0.5)
                }
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._fallback_response(symbolic_decision)
                
        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return self._fallback_response(symbolic_decision)
    
    def _create_validation_prompt(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> str:
        """Create validation prompt"""
        
        return f"""You are an expert data preprocessing consultant. Evaluate this preprocessing decision for QUALITY and REAL-WORLD applicability.

COLUMN INFORMATION:
- Name: {column_info['name']}
- Data Type: {column_info['dtype']}
- Rows: {column_info['row_count']}
- Null %: {column_info['null_pct']:.1%}
- Unique values: {column_info['unique_count']} ({column_info['unique_ratio']:.1%})
- Sample values: {column_info['sample_values']}

STATISTICS:
- Skewness: {column_info.get('skewness', 'N/A')}
- Outliers: {column_info.get('outlier_pct', 'N/A')}
- Entropy: {column_info.get('entropy', 'N/A')}

SYMBOLIC ENGINE DECISION:
- Action: {symbolic_decision['action']}
- Confidence: {symbolic_decision['confidence']:.1%}
- Reasoning: {symbolic_decision['explanation']}

EVALUATION CRITERIA:
1. Data Quality: Does this preserve important information?
2. ML Compatibility: Will this work well with ML models?
3. Real-World: Is this practical for production use?
4. Edge Cases: Does this handle unusual patterns correctly?

TASK: Is this decision CORRECT for a quality data preprocessing tool?

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
    "is_correct": true or false,
    "correct_action": "action_name" (if incorrect, otherwise same as input),
    "reasoning": "brief explanation focusing on quality and real-world use (max 100 words)",
    "confidence": 0.0 to 1.0
}}"""

    def _fallback_response(self, symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback if LLM fails - trust symbolic"""
        return {
            'is_correct': True,
            'correct_action': symbolic_decision['action'],
            'reasoning': 'LLM unavailable, trusting symbolic engine',
            'llm_confidence': 0.5
        }
    
    def test_connection(self) -> bool:
        """Test if LLM is available"""
        try:
            if self.mode == "gemini":
                response = self.gemini_model.generate_content("Test")
                return True
            elif self.mode == "huggingface":
                response = self.hf_client.text_generation(
                    "Test",
                    model=self.hf_model,
                    max_new_tokens=10
                )
                return True
            else:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model, "prompt": "Test", "stream": False},
                    timeout=10
                )
                return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("LLM CLIENT TEST")
    print("="*60)
    
    # Test Hugging Face
    print("\n🤗 Testing Hugging Face (FREE)...")
    hf_token = input("Enter Hugging Face token (or press Enter to skip): ").strip()
    
    if hf_token:
        try:
            client_hf = LLMClient(mode="huggingface", api_key=hf_token)
            if client_hf.test_connection():
                print("✅ Hugging Face is available!")
                
                # Test validation
                column_info = {
                    'name': 'Year',
                    'dtype': 'int64',
                    'row_count': 1000,
                    'null_pct': 0.0,
                    'unique_count': 50,
                    'unique_ratio': 0.05,
                    'sample_values': [2020, 2019, 2021],
                    'skewness': 0.1
                }
                
                symbolic_decision = {
                    'action': 'standard_scale',
                    'confidence': 0.75,
                    'explanation': 'Numeric column'
                }
                
                result = client_hf.validate_decision(column_info, symbolic_decision)
                print(f"\n✅ Validation test:")
                print(f"   Is correct: {result['is_correct']}")
                print(f"   Correct action: {result['correct_action']}")
                print(f"   Reasoning: {result['reasoning']}")
            else:
                print("❌ Hugging Face not available")
        except Exception as e:
            print(f"❌ Hugging Face test failed: {e}")
    else:
        print("⏭️  Skipped Hugging Face test")
    
    print("\n" + "="*60)
    print("Get Hugging Face token:")
    print("https://huggingface.co/settings/tokens")
    print("="*60)
