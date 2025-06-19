"""
OpenAI API client for accommodation speech detection
"""

import os
from typing import Optional


class OpenAIClient:
    """OpenAI API client wrapper"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Try to load from .env file
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            self.api_key = line.split('=', 1)[1].strip()
                            break
            except:
                pass
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Try to use OpenAI package, fallback to requests
        self.use_package = False
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_package = True
        except ImportError:
            self.client = None
    
    def predict(self, text: str, prompt: str, model: str = "gpt-4") -> str:
        """
        Get prediction from OpenAI
        
        Args:
            text: Input text to evaluate
            prompt: System prompt
            model: Model to use
            
        Returns:
            Prediction: "0" or "1"
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        
        if self.use_package and self.client:
            # Use OpenAI package
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI package error: {e}, falling back to requests")
                self.use_package = False
        
        # Fallback to requests
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API Error: {e}")
            return "0"