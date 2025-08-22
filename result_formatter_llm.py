import google.generativeai as genai
import os
from typing import Optional, List, Tuple
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Gemini model with authentication"""
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"

from typing import List, Tuple, Optional

class ResultFormatterLLM(GeminiLLM):
    def __init__(self):
        super().__init__()
    
    def format_query_results(self, 
                           user_query: str, 
                           sql_query: str,
                           columns: Optional[List[str]], 
                           results: List[Tuple],
                           error_message: Optional[str] = None,
                           total_rows: Optional[int] = None) -> str:
        
        if error_message:
            prompt = f"""The user asked: "{user_query}"

There was an error processing the request.

Please provide a brief, professional response explaining that the request couldn't be processed and suggest how the user might rephrase their question. Be helpful and encouraging. Do not mention any technical details, SQL queries, or database tables.

Response:"""
            
            return self.generate_response(prompt, temperature=0)
        
        # Use total_rows if provided, otherwise fall back to len(results)
        actual_total_rows = total_rows if total_rows is not None else len(results)
        
        # Handle successful queries with no results
        if actual_total_rows == 0:
            prompt = f"""The user asked: "{user_query}"

No data was found matching the request.

Please provide a brief response (2-3 sentences max) that:
1. States that no data was returned
2. Provides 3-4 practical suggestions for what the user can modify or try differently
3. Keep it concise and user-friendly
4. Do not mention SQL queries, database tables, or any technical implementation details

Response:"""
            
            return self.generate_response(prompt, temperature=0.3)
        
        # Handle successful queries with results
        # Prepare the data for the prompt
        data_sample = self._prepare_data_sample(columns, results, actual_total_rows)
        
        prompt = f"""User asked: "{user_query}"

Found {actual_total_rows} records. Here's a sample:

{data_sample}

1. Answer the user's question in simple, natural language. Be brief and direct.
2. Present the information in an easy-to-read format
3. Use natural language that a business user would understand
4. Be concise but informative
5. Keep the tone professional and helpful

Response:"""
        
        return self.generate_response(prompt, temperature=0.2)
    
    def _prepare_data_sample(self, columns: List[str], results: List[Tuple], total_rows: int, max_rows: int = 10) -> str:
        """
        Prepare a formatted sample of the data for the LLM prompt
        
        Args:
            columns (List[str]): Column names
            results (List[Tuple]): Query results (may be truncated)
            total_rows (int): Total number of rows in the actual result set
            max_rows (int): Maximum number of rows to include in sample
            
        Returns:
            str: Formatted data sample
        """
        if not results:
            return "No data available"
        
        # Take a sample of the data
        sample_results = results[:max_rows]
        
        # Format as a simple table
        formatted_data = []
        
        # Add header
        header = " | ".join(columns)
        formatted_data.append(header)
        formatted_data.append("-" * len(header))
        
        # Add rows
        for row in sample_results:
            formatted_row = " | ".join(
                str(value) if value is not None else "NULL" 
                for value in row
            )
            formatted_data.append(formatted_row)
        
        # Add indicator if there are more rows
        if total_rows > len(sample_results):
            formatted_data.append(f"... and {total_rows - len(sample_results)} more records")
        
        return "\n".join(formatted_data)
