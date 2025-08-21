import mysql.connector
from typing import Tuple, List, Optional, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Database connection parameters - can be configured via environment variables
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'MYSQL'),
            'database': os.getenv('DB_NAME', 'evapharma')
        }
    
    def execute_query(self, query: str) -> Tuple[Optional[List[str]], Any]:
        """
        Execute SQL query and return results
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Tuple[Optional[List[str]], Any]: (columns, results) where:
                - columns: List of column names (for SELECT queries) or None
                - results: Query results (rows for SELECT, success message for others, or error message)
        """
        conn = None
        cursor = None
        
        try:
            self.logger.info(f"Connecting to database: {self.db_config['host']}")
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            self.logger.info(f"Executing query: {query[:100]}...")  # Log first 100 chars
            cursor.execute(query)
            
            # Handle SELECT queries
            if query.strip().lower().startswith("select"):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                self.logger.info(f"Query returned {len(results)} rows")
                return columns, results
            
            # Handle INSERT, UPDATE, DELETE queries
            else:
                conn.commit()
                affected_rows = cursor.rowcount
                self.logger.info(f"Query executed successfully, {affected_rows} rows affected")
                return None, f"Query executed successfully. {affected_rows} rows affected."
                
        except mysql.connector.Error as err:
            self.logger.error(f"Database error: {err}")
            return None, f"Database Error: {err}"
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None, f"Unexpected Error: {e}"
            
        finally:
            # Clean up connections
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            conn.close()
            self.logger.info("Database connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def format_results_for_display(self, columns: List[str], results: List[Tuple]) -> str:
        """
        Format query results in a readable table format
        
        Args:
            columns (List[str]): Column names
            results (List[Tuple]): Query results
            
        Returns:
            str: Formatted table string
        """
        if not results:
            return "No data found."
        
        # Calculate column widths
        col_widths = [len(col) for col in columns]
        
        for row in results:
            for i, value in enumerate(row):
                str_value = str(value) if value is not None else "NULL"
                col_widths[i] = max(col_widths[i], len(str_value))
        
        # Create header
        header = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        separator = "-" * len(header)
        
        # Create rows
        rows = []
        for row in results:
            formatted_row = " | ".join(
                str(value).ljust(col_widths[i]) if value is not None else "NULL".ljust(col_widths[i])
                for i, value in enumerate(row)
            )
            rows.append(formatted_row)
        
        return f"{header}\n{separator}\n" + "\n".join(rows)
    
    def get_result_summary(self, columns: List[str], results: List[Tuple]) -> dict:
        """
        Get summary statistics about the query results
        
        Args:
            columns (List[str]): Column names
            results (List[Tuple]): Query results
            
        Returns:
            dict: Summary information
        """
        return {
            "total_rows": len(results),
            "total_columns": len(columns),
            "column_names": columns,
            "has_data": len(results) > 0
        }