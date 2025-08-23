import google.generativeai as genai
import os
from typing import Optional
from dotenv import load_dotenv
import requests

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

    def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
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

class OllamaMistralLLM:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response from Ollama's local Mistral model.

        Args:
            prompt (str): Input prompt
            temperature (float): Sampling temperature
        Returns:
            str: Generated text
        """
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature
                },
                "stream": False
            }

            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Ollama returns JSON with 'response' for the generated text
            return data.get("response", "").strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"



class QwenLLM:
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Placeholder for symmetry with GeminiLLM — Ollama loads models on request."""
        try:
            # For Ollama, no explicit preload; ensure server is reachable
            resp = requests.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            raise

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response from the Ollama model.

        Args:
            prompt (str): Input prompt
            temperature (float): Sampling temperature

        Returns:
            str: Generated response
        """
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature
                },
                "stream": False
            }

            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            return data.get("response", "").strip()

        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return "Error generating response"

class InjectionCheckLLM(GeminiLLM):
    def __init__(self):
        super().__init__()

    def check_injection(self, user_input: str) -> str:
        """
        Check if user input is valid, injection risk, or unrelated to database queries

        Args:
            user_input (str): User's natural language query

        Returns:
            str: "valid", "injection", or "unrelated"
        """
        injection_check_prompt = """
You are a security guard and query classifier for a healthcare analytics database system.

DATABASE CONTEXT:
- Database name: healthcare_analytics
- Main tables: july_HCPs (healthcare professionals), july_interactions (medical representative interactions)
- july_HCPs contains: id (PK), customerid, englishname, isconsultant, isdecisionmaker, issamspeaker, isuniversitystaff, isampmspeaker, customerclassificationid, CustomerClassification, specialityid, Speciality, countryid, Country
  Any query asking about HCP IDs, customer information, names, consultant status, decision maker status, speaker roles, university affiliation, classifications, specialties, or countries is valid.

- july_interactions contains: MRId, MRArFullName, InteractionId (PK), InteractionStatusId, InteractionStatus, reportdate, lineid, LineName, businessUnitId, BusinessUnitName, HCPId (FK), HCPCustomerId, HCPEnglishName, HCPArabicName, SpecialtyId, Specialty
  Any query about medical representative IDs, names, interaction details, statuses, dates, line information, business units, or HCP-related data is valid.

Your job: Classify the user's input into exactly one of these three categories:

1. **VALID** - Return "valid" for legitimate database queries about:
   - Healthcare professionals (HCPs): names, specialties, countries, classifications, roles
   - Medical representative interactions: counts, dates, status, business units
   - Data analysis: filtering, grouping, counting, listing, aggregating
   - Legitimate business questions about the healthcare data

2. **INJECTION** - Return "injection" for SQL injection risks or database manipulation:
   - SQL commands: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE
   - SQL injection patterns: OR 1=1, UNION SELECT, ; --, /* */, xp_cmdshell
   - Database modification attempts
   - Security bypass attempts
   - Any attempt to change, delete, or manipulate database structure/data

3. **UNRELATED** - Return "unrelated" for queries completely unrelated to healthcare analytics database:
   - Weather, jokes, translations, mathematics, cooking, music, entertainment
   - General knowledge questions unrelated to healthcare/database
   - Personal questions about company executives or staff
   - Technical support for non-database systems
   - Requests for external services (Spotify, etc.)
   - Creative writing, storytelling, or fiction requests

Examples:

VALID queries (return "valid"):
- "List all HCPs from Egypt with cardiology specialty"
- "Count interactions by medical representative"
- "Show HCPs who are consultants and decision makers"
- "Find interactions with approved status in July 2024"
- "Get Arabic names of HCPs"

INJECTION queries (return "injection"):
- "DROP TABLE july_HCPs"
- "SELECT * FROM users WHERE 1=1 OR 'a'='a'"
- "DELETE FROM july_interactions WHERE true"
- "; DROP DATABASE healthcare_analytics; --"

UNRELATED queries (return "unrelated"):
- "What is the weather today in Cairo?"
- "Tell me a joke about doctors"
- "Translate this sentence to French"
- "Who is the president of EVA Pharma?"
- "What's 5+5?"
- "How do I cook pasta?"
- "Play music from Spotify"

User input: {user_input}

Classification:"""

        try:
            response = self.generate_response(
                injection_check_prompt.format(user_input=user_input),
                temperature=0.0
            )

            # Clean the response and check for classification
            response = response.strip().lower()

            # Handle various possible responses
            if "valid" in response:
                return "valid"
            elif "injection" in response:
                return "injection"
            elif "unrelated" in response:
                return "unrelated"
            else:
                # If unclear response, err on the side of caution
                print(f"Unclear injection check response: {response}")
                return "unrelated"

        except Exception as e:
            print(f"Error in injection check: {e}")
            # If there's an error, err on the side of caution
            return "unrelated"


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

        return self.generate_response(prompt, temperature=0)


class SQLGeneratorLLM(GeminiLLM):
    def __init__(self):
        super().__init__()

    def generate_sql(self, query: str, reasoning: str, schema_context: str) -> str:
        """
        Generate SQL query based on user query and schema using strict SQL generation rules.
        """
        prompt = f"""You are an expert SQL generator. Given step-by-step reasoning describing how to build an SQL query, generate the exact MySQL query.

Follow these strict HARD RULES (must follow exactly):
- Use only the tables and columns listed in the schema.
- Use proper JOINs, WHERE clauses, GROUP BY, ORDER BY, DISTINCT, and aggregations as described.
- For boolean columns listed in the schema, always compare using = TRUE or = FALSE (never 1/0).
- Output only a single valid SQL query — no explanations or extra text.
- Escape single quotes in string literals.
- Use MySQL syntax.
- Only add JOINs if the requested columns come from different tables per the provided schema relationships.
- Prefer the simplest single-table query when the requested columns exist in one table.
- If the reasoning is ambiguous about tables or columns, do your best to infer from the schema.
- If the user says "show all or find all" (and does NOT name a column), use: SELECT * FROM <table> ...
- Never expand SELECT * into individual columns unless the user explicitly asks to list columns.
- Do NOT add DISTINCT unless the user explicitly asks for “unique”, “distinct”, or “no duplicates”.
- Do NOT add ORDER BY unless the user explicitly asks for sorting.
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

        response = self.generate_response(prompt, temperature=0.0)
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

        response = self.generate_response(prompt, temperature=0.0)
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
                    response = response[:i + 1]
                    break

            sql_lines = [response.strip()]

        # Join the SQL lines
        corrected_sql = ' '.join(sql_lines) if sql_lines else response.strip()

        # Ensure it ends with semicolon if it doesn't already
        if corrected_sql and not corrected_sql.endswith(';'):
            corrected_sql += ';'

        return corrected_sql