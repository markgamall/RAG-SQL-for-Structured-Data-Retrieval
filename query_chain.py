from typing import Dict, Any
import logging
import sqlparse
from rag_retriever import RAGRetriever
from llm_models import ReasoningLLM, SQLGeneratorLLM, SQLCorrectorLLM, InjectionCheckLLM


class QueryToSQLChain:
    def __init__(self, top_k: int = 2, persist_directory: str = "./chroma_db"):
        self.rag_retriever = RAGRetriever(persist_directory=persist_directory)
        self.injection_check_llm = InjectionCheckLLM()
        self.reasoning_llm = ReasoningLLM()
        self.sql_generator_llm = SQLGeneratorLLM()
        self.sql_corrector_llm = SQLCorrectorLLM() 
        self.top_k = top_k
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_sql_syntax(self, sql: str) -> bool:
        """Validate SQL syntax using sqlparse"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False  # Empty or invalid
            # Check if the parsed result contains actual SQL statements
            for statement in parsed:
                if statement.tokens:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"SQL validation error: {e}")
            return False

    def process_query(self, user_query: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing query: {user_query}")
            
            # Step 0: Check for SQL injection before processing
            self.logger.info("Step 0: Checking for SQL injection risks...")
            is_safe = self.injection_check_llm.check_injection(user_query)
            
            if not is_safe:
                self.logger.warning(f"Potentially unsafe query detected: {user_query}")
                return {
                    "status": "error",
                    "error_type": "security_violation",
                    "error_message": "This query contains inappropriate content that violates security policies. Please rephrase your request using only standard data retrieval language without any database modification commands.",
                    "user_query": user_query
                }
            
            self.logger.info("Query passed security check, continuing with processing...")
            
            # Step 1: Retrieve relevant schema chunks using RAG
            self.logger.info("Step 1: Retrieving schema chunks...")
            schema_context = self.rag_retriever.get_schema_context(user_query, self.top_k)
            retrieved_chunks = self.rag_retriever.retrieve_chunks(user_query, self.top_k)
            
            self.logger.info(f"Retrieved {len(retrieved_chunks)} relevant schema chunks")
            
            # Step 2: Generate reasoning using first LLM
            self.logger.info("Step 2: Generating reasoning...")
            reasoning = self.reasoning_llm.generate_reasoning(user_query, schema_context)
            
            print("Reasoning:", reasoning)
            self.logger.info("Reasoning generated successfully")
            
            # Step 3: Generate SQL using second LLM
            self.logger.info("Step 3: Generating SQL query...")
            sql_query = self.sql_generator_llm.generate_sql(user_query, reasoning, schema_context)
            
            print("Initial SQL query:", sql_query)
            self.logger.info("Initial SQL query generated")
            
            # Step 4: Validate and correct SQL if needed
            is_valid_sql = self.validate_sql_syntax(sql_query)
            
            if is_valid_sql:
                valid_sql_query = sql_query
                self.logger.info("SQL query is valid")
            else:
                self.logger.info("SQL query is invalid, attempting correction...")
                print("Invalid SQL detected, correcting...")
                
                # Use the corrector LLM to fix the SQL
                valid_sql_query = self.sql_corrector_llm.correct_sql(
                    invalid_sql=sql_query,
                    schema_context=schema_context,
                    user_query=user_query
                )
                
                print("Corrected SQL query:", valid_sql_query)
                
                # Validate the corrected SQL
                is_corrected_valid = self.validate_sql_syntax(valid_sql_query)
                if is_corrected_valid:
                    self.logger.info("SQL query corrected successfully")
                else:
                    self.logger.warning("SQL correction failed, using original query")
                    # If correction fails, we'll still return the corrected attempt
                    # but mark it in the response
            
            # Prepare response
            result = {
                "status": "success",
                "user_query": user_query,
                "retrieved_chunks": [
                    {
                        "content": chunk[0],
                        "relevance_score": chunk[1]
                    }
                    for chunk in retrieved_chunks
                ],
                "schema_context": schema_context,
                "reasoning": reasoning,
                "original_sql": sql_query,
                "sql_query": valid_sql_query,
                "was_corrected": not is_valid_sql,
                "is_valid": self.validate_sql_syntax(valid_sql_query),
                "security_check_passed": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "user_query": user_query
            }
    
    def get_sql_only(self, user_query: str) -> str:
        """Get only the SQL query result"""
        result = self.process_query(user_query)
        
        if result["status"] == "success":
            return result["sql_query"]
        else:
            return f"Error: {result.get('error_message', 'Unknown error')}"
    
    def get_detailed_response(self, user_query: str) -> Dict[str, Any]:
        """Get detailed response with all processing steps"""
        return self.process_query(user_query)


def create_query_chain(top_k: int = 2, persist_directory: str = "./chroma_db") -> QueryToSQLChain:
    """Factory function to create a QueryToSQLChain instance"""
    return QueryToSQLChain(top_k=top_k, persist_directory=persist_directory)