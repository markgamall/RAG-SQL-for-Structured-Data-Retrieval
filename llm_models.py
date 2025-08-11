import google.generativeai as genai
import os
from typing import Optional
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
        """
        Generate response from the model
        
        Args:
            prompt (str): Input prompt
            temperature (float): Temperature for sampling
            
        Returns:
            str: Generated response
        """
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


class InjectionCheckLLM(GeminiLLM):
    def __init__(self):
        super().__init__()
    
    def check_injection(self, user_input: str) -> bool:
        """
        Check if user input contains SQL injection risks
        
        Args:
            user_input (str): User's natural language query
            
        Returns:
            bool: True if safe, False if potentially dangerous
        """
        injection_check_prompt = """
You are a security guard for SQL inputs.
Your job: Check if the user's natural language input contains any SQL injection risks or suspicious patterns.
- If the input is safe and contains no SQL injection risk, return "True"
- If you detect any risk or suspicious content that could lead to SQL injection, return "False"

Examples of unsafe inputs (return False):
- DROP TABLE users;
- SELECT * FROM users WHERE username = 'admin' OR 1=1
- DELETE FROM orders WHERE 'a'='a' OR '1'='1
- any deletion, truncation or dropping or altering or updating or modifying anything in the database or inserting, anything that changes the database

Examples of safe inputs (return True):
- List all customers with country = 'USA'
- Show me interactions from July 2023
- Get the Arabic names of HCPs who had 'Approved' status

User input:
{user_input}

Output:"""

        try:
            response = self.generate_response(
                injection_check_prompt.format(user_input=user_input), 
                temperature=0.0
            )
            
            # Clean the response and check for True/False
            response = response.strip().lower()
            
            # Handle various possible responses
            if "true" in response:
                return True
            elif "false" in response:
                return False
            else:
                # If unclear response, err on the side of caution
                print(f"Unclear injection check response: {response}")
                return False
                
        except Exception as e:
            print(f"Error in injection check: {e}")
            # If there's an error, err on the side of caution
            return False


class ReasoningLLM(GeminiLLM):
    def __init__(self):
        super().__init__()
    
    def generate_reasoning(self, query: str, schema_context: str) -> str:
        
        prompt = f"""You are an expert SQL reasoning assistant. Your job is to analyze the user's natural language question and database schema, then explain step-by-step how to translate the question into an SQL query.

Given the user question and the database schema, think through:

- Which tables and columns are involved?
- What filters, conditions, and joins are required?
- What aggregation, grouping, ordering, or limits are necessary?

Explain your reasoning clearly and list the steps the SQL query should follow.

Example 1:

User question: "List the Arabic names of HCPs who had interactions with 'Approved' status."

Schema:
Table: july_interactions(MRId, MRArFullName, InteractionId, InteractionStatusId, InteractionStatus, reportdate, lineid, 
LineName, businessUnitId, BusinessUnitName, HCPId, HCPCustomerId, HCPEnglishName, HCPArabicName, SpecialtyId, Specialty)
Table: july_HCPs(id, customerid, englishname, isconsultant, isdecisionmaker, issamspeaker, 
isuniversitystaff, isampmspeaker, customerclassificationid, CustomerClassification, specialityid, 
Speciality, countryid, Country)

Reasoning:
1. Identify the tables involved: 'july_interactions' and 'july_HCPs'.
2. Join 'july_interactions' with 'july_HCPs' on HCPId = id.
3. Filter 'july_interactions' where InteractionStatus = 'Approved'.
4. Select distinct 'HCPArabicName' from the joined tables.

---

Example 2:

User question: "Find all HCP English names with Specialty ID 5 who had interactions after July 1, 2025."

Schema:
Table: july_interactions(MRId, MRArFullName, InteractionId, InteractionStatusId, InteractionStatus, reportdate, lineid, 
LineName, businessUnitId, BusinessUnitName, HCPId, HCPCustomerId, HCPEnglishName, HCPArabicName, SpecialtyId, Specialty)
Table: july_HCPs(id, customerid, englishname, isconsultant, isdecisionmaker, issamspeaker, 
isuniversitystaff, isampmspeaker, customerclassificationid, CustomerClassification, specialityid, 
Speciality, countryid, Country)

Reasoning:
1. Use tables 'july_interactions' and 'july_HCPs'.
2. Join on HCPId = id.
3. Filter 'july_HCPs' where specialityid = 5.
4. Filter 'july_interactions' where reportdate > '2025-07-01'.
5. Select distinct 'englishname' from the joined tables.

---

Example 3:

User question: "Count how many interactions each Business Unit had in June 2025."

Schema:
Table: july_interactions(MRId, MRArFullName, InteractionId, InteractionStatusId, InteractionStatus, reportdate, lineid, 
LineName, businessUnitId, BusinessUnitName, HCPId, HCPCustomerId, HCPEnglishName, HCPArabicName, SpecialtyId, Specialty)
Table: july_HCPs(id, customerid, englishname, isconsultant, isdecisionmaker, issamspeaker, 
isuniversitystaff, isampmspeaker, customerclassificationid, CustomerClassification, specialityid, 
Speciality, countryid, Country)

Reasoning:
1. Use table 'july_interactions'.
2. Filter 'reportdate' to dates in June 2025.
3. Group by 'businessUnitId' and 'BusinessUnitName'.
4. Count the number of 'InteractionId' per business unit.
5. Select business unit name and interaction count.

---

User question: {query}

Schema:
{schema_context}

Reasoning:"""
        
        return self.generate_response(prompt, temperature=0.3)


class SQLGeneratorLLM(GeminiLLM):
    def __init__(self):
        super().__init__()
    
    def generate_sql(self, query: str, reasoning: str, schema_context: str) -> str:
        """
        Generate SQL query based on user query and schema using strict SQL generation rules.
        """
        prompt = f"""You are an expert SQL generator. Given step-by-step reasoning describing how to build an SQL query, generate the exact MySQL query.

Follow these rules:
- Use only the tables and columns listed in the schema.
- Use proper JOINs, WHERE clauses, GROUP BY, ORDER BY, DISTINCT, and aggregations as described.
- Output only a single valid SQL query — no explanations or extra text.
- Escape single quotes in string literals.
- Use MySQL syntax.
- If the reasoning is ambiguous about tables or columns, do your best to infer from the schema.
- Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
- Pay attention to use only the column names that you can see in the schema description.
- Be careful to not query for columns that do not exist.
- Pay attention to which column is in which table.

Schema:
{schema_context}

Reasoning steps:
{reasoning}

Query:
{query}

SQL:
"""

        response = self.generate_response(prompt, temperature=0.1)
        return self._extract_sql(response)

    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from the response"""
        # Remove common prefixes and clean up
        response = response.strip()
        
        # Remove markdown formatting if present
        if response.startswith("```sql"):
            response = response.replace("```sql", "").replace("```", "").strip()
        elif response.startswith("```"):
            response = response.replace("```", "").strip()
        
        # Ensure it starts with SELECT if it's a query
        if not response.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
            if "SELECT" in response.upper():
                # Find the first SELECT and start from there
                select_index = response.upper().find("SELECT")
                response = response[select_index:]
        
        return response.strip()


class SQLCorrectorLLM(GeminiLLM):
    def __init__(self):
        super().__init__()
    
    def correct_sql(self, invalid_sql: str, schema_context: str, user_query: str = "") -> str:
        prompt = f"""You are an expert SQL corrector. Your job is to fix the syntax errors in the given SQL query and make it valid MySQL syntax.

Rules:
- Fix syntax errors, missing keywords, incorrect punctuation, misplaced parentheses
- Ensure proper MySQL syntax
- Use only tables and columns from the provided schema
- Output ONLY the corrected SQL query - no explanations, no extra text, no markdown
- Do not change the logic or intent of the query, only fix syntax issues
- Ensure proper JOIN syntax, WHERE clauses, quotes, semicolons, etc.

Schema:
{schema_context}

Original user question (for context): {user_query}

Invalid SQL query to fix:
{invalid_sql}

Corrected SQL:"""

        response = self.generate_response(prompt, temperature=0.1)
        return self._extract_sql_clean(response)
    
    def _extract_sql_clean(self, response: str) -> str:
        """Extract and clean SQL query from response, ensuring only SQL is returned"""
        response = response.strip()
        
        # Remove markdown formatting
        if "```sql" in response:
            response = response.split("```sql")[1].split("```")[0].strip()
        elif "```" in response:
            # Handle generic code blocks
            parts = response.split("```")
            if len(parts) >= 2:
                response = parts[1].strip()
        
        # Remove any explanatory text before or after SQL
        lines = response.split('\n')
        sql_lines = []
        sql_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start collecting when we see SQL keywords
            if line.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                sql_started = True
            
            # If SQL has started, collect the line
            if sql_started:
                sql_lines.append(line)
                
                # Stop if we hit a semicolon at the end of a line (end of query)
                if line.endswith(';'):
                    break
        
        # If no SQL keywords found, try to find SELECT in the text
        if not sql_lines and 'SELECT' in response.upper():
            select_index = response.upper().find('SELECT')
            response = response[select_index:]
            
            # Find the end of the query (semicolon or end of reasonable SQL)
            for i, char in enumerate(response):
                if char == ';':
                    response = response[:i+1]
                    break
            
            sql_lines = [response.strip()]
        
        # Join the SQL lines
        corrected_sql = ' '.join(sql_lines) if sql_lines else response.strip()
        
        # Ensure it ends with semicolon if it doesn't already
        if corrected_sql and not corrected_sql.endswith(';'):
            corrected_sql += ';'
        
        return corrected_sql