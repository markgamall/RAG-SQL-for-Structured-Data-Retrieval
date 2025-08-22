import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="EVAInteract",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:5000"  # Change this to your API URL

class EVAAnalyticsAPI:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=None)
            return {"status": "success", "data": response.json()}
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def query_database(self, query: str) -> Dict[str, Any]:
        """Send query to the database and get natural language response with table data"""
        try:
            payload = {"query": query}
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=None
            )
            
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {"status": "error", "data": error_data}
                
        except requests.RequestException as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def test_database_connection(self) -> Dict[str, Any]:
        """Test database connectivity"""
        try:
            response = requests.get(f"{self.base_url}/db/test", timeout=None)
            return {"status": "success", "data": response.json()}
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api" not in st.session_state:
        st.session_state.api = EVAAnalyticsAPI()
    
    if "api_status" not in st.session_state:
        st.session_state.api_status = None

def check_api_status():
    """Check and display API status"""
    if st.session_state.api_status is None:
        with st.spinner("Checking API connection..."):
            health_result = st.session_state.api.health_check()
            db_result = st.session_state.api.test_database_connection()
            
            st.session_state.api_status = {
                "api_healthy": health_result["status"] == "success",
                "db_connected": db_result["status"] == "success",
                "last_checked": datetime.now()
            }
    
    return st.session_state.api_status

def display_table_data(table_data):
    """Display table data in a clean format"""
    if table_data and table_data["columns"] and table_data["rows"]:
        df = pd.DataFrame(table_data["rows"], columns=table_data["columns"])
        
        # Show row count info
        row_count = table_data["row_count"]
        displayed_rows = len(table_data["rows"])
        
        if table_data.get("has_more_data", False):
            st.caption(f"Showing first {displayed_rows} of {row_count} total records")
        else:
            st.caption(f"Total records: {row_count}")
        
        # Display the table
        st.dataframe(df, use_container_width=True, height=min(400, max(200, len(df) * 35 + 60)))
        
        return df
    else:
        st.info("No data to display")
        return None

def display_query_details(query_details):
    """Display query details in an expandable section"""
    with st.expander("Query Details", expanded=False):
        if "sql_query" in query_details:
            st.markdown("**Generated SQL Query:**")
            st.code(query_details["sql_query"], language="sql")
        
        # Show reasoning output - this field comes from the API as part of the detailed response
        if "reasoning" in query_details:
            st.markdown("**Reasoning Output:**")
            st.text(query_details["reasoning"])
        
        # Show injection check result - need to add this to the API response
        if "injection_check_result" in query_details:
            st.markdown("**Injection Check Result:**")
            st.text(query_details["injection_check_result"])
        
        if "was_sql_corrected" in query_details:
            if query_details["was_sql_corrected"]:
                st.info("SQL query was automatically corrected")
        
        if "processing_summary" in query_details:
            st.markdown("**Processing Summary:**")
            st.text(query_details["processing_summary"])
        
        if "error_details" in query_details:
            st.error(f"Error Details: {query_details['error_details']}")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("EVAInteract")
    st.write(
        "Welcome to EVAInteract! I’m here to help you access, query, and analyze your data using simple natural language."
    )
  
  
    # Sidebar
    with st.sidebar:
        st.logo("eva_logo_y.png", size="large")
        st.header("Settings")

        st.session_state.show_details = st.toggle(
            "Show Query Details", 
            value=False,
            help="Display SQL queries and technical details"
        )
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Check API status silently
    status_info = check_api_status()
    
    # Main chat interface
    if not status_info["api_healthy"]:
        st.error("Cannot connect to EVA Analytics API. Please check if the API server is running.")
        st.info("Make sure your Flask API is running on http://localhost:5000")
        return

    # Initialize messages with welcome message if not present
    if len(st.session_state.messages) == 0:
        welcome_message = {
            "role": "assistant", 
            "content": """Hey there! What’s on your mind today?"""
        }
        st.session_state.messages.append(welcome_message)

    # Chat interface
    st.subheader("Chat")
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display metadata if available and show_details is enabled
            if message["role"] == "assistant" and "metadata" in message and st.session_state.show_details:
                metadata = message["metadata"]
            
            # Display table data if available
            if message["role"] == "assistant" and "table_data" in message:
                if message["table_data"]:
                    st.markdown("### Results")
                    display_table_data(message["table_data"])
            
            # Display query details if available and show_details is enabled
            if message["role"] == "assistant" and "query_details" in message and st.session_state.show_details:
                display_query_details(message["query_details"])

    # Handle predefined message selection
    chat_placeholder = "Type your question here or use the quick questions above..."

    if "selected_message" in st.session_state:
        default_value = st.session_state.selected_message
        del st.session_state.selected_message  
    else:
        default_value = ""

    if prompt := st.chat_input(chat_placeholder):
        user_message = prompt
    elif default_value:
        user_message = default_value
    else:
        user_message = None

    if user_message:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.chat_message("assistant"):
            with st.spinner("EVA Assistant is thinking..."):
                result = st.session_state.api.query_database(user_message)
            
            if result["status"] == "success":
                data = result["data"]
                
                if data.get("status") == "success":
                    # Display the formatted response
                    formatted_response = data.get("formatted_response", "Query processed successfully!")
                    st.markdown(formatted_response)
                    
                    # Display table data
                    table_data = data.get("table_data")
                    if table_data:
                        st.markdown("### Results")
                        display_table_data(table_data)
                    
                    # Add to session state
                    assistant_message = {
                        "role": "assistant",
                        "content": formatted_response,
                        "table_data": table_data
                    }
                    
                    # Add query details if available
                    if "query_details" in data:
                        assistant_message["query_details"] = data["query_details"]
                    
                    # Display query details if enabled
                    if st.session_state.show_details and "query_details" in data:
                        display_query_details(data["query_details"])
                    
                    st.session_state.messages.append(assistant_message)
                    
                else:
                    # Error case with formatted response
                    error_msg = data.get("formatted_response", data.get("message", "Unknown error occurred"))
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "error", 
                        "content": error_msg,
                        "query_details": data.get("query_details", {})
                    })
            
            else:
                error_msg = result.get("message", "Failed to process query")
                if "data" in result:
                    if "formatted_response" in result["data"]:
                        error_msg = result["data"]["formatted_response"]
                    elif "message" in result["data"]:
                        error_msg = result["data"]["message"]
                
                st.error(error_msg)
                st.session_state.messages.append({"role": "error", "content": error_msg})
        
        if default_value:
            st.rerun()

if __name__ == "__main__":
    main()
