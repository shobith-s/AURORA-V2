"""
LLM Client with support for both Local (Ollama) and Cloud (Gemini API)
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM validation - supports local Ollama and Google Gemini"""
    
    def __init__(
        self, 
        mode: Literal["local", "gemini"] = "local",
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
        else:
            logger.info(f"✅ Using {model} (local)")
        
    def validate_decision(self, column_info: Dict[str, Any], symbolic_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing decision using LLM"""
        
        if self.mode == "gemini":
            return self._validate_with_gemini(column_info, symbolic_decision)
        else:
            return self._validate_with_ollama(column_info, symbolic_decision)
    
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
    
    # Test both modes
    print("="*60)
    print("LLM CLIENT TEST")
    print("="*60)
    
    # Test local
    print("\n1. Testing LOCAL mode (Ollama)...")
    try:
        client_local = LLMClient(mode="local")
        if client_local.test_connection():
            print("✅ Local Ollama is available")
        else:
            print("❌ Local Ollama not available")
    except Exception as e:
        print(f"❌ Local test failed: {e}")
    
    # Test Gemini
    print("\n2. Testing GEMINI mode...")
    api_key = input("Enter Gemini API key (or press Enter to skip): ").strip()
    
    if api_key:
        try:
            client_gemini = LLMClient(mode="gemini", api_key=api_key)
            if client_gemini.test_connection():
                print("✅ Gemini API is available")
                
                # Test validation
                column_info = {
                    'name': 'Year',
                    'dtype': 'int64',
                    'row_count': 1000,
                    'null_pct': 0.0,
                    'unique_count': 50,
                    'unique_ratio': 0.05,
                    'sample_values': [2020, 2019, 2021, 2020, 2022],
                    'skewness': 0.1
                }
                
                symbolic_decision = {
                    'action': 'standard_scale',
                    'confidence': 0.75,
                    'explanation': 'Numeric column with moderate range'
                }
                
                result = client_gemini.validate_decision(column_info, symbolic_decision)
                print(f"\n✅ Validation test:")
                print(f"   Is correct: {result['is_correct']}")
                print(f"   Correct action: {result['correct_action']}")
                print(f"   Reasoning: {result['reasoning']}")
            else:
                print("❌ Gemini API not available")
        except Exception as e:
            print(f"❌ Gemini test failed: {e}")
    else:
        print("⏭️  Skipped Gemini test")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("- Local: Free, private, but needs good hardware")
    print("- Gemini: Free, better quality, works on any hardware")
    print("="*60)
