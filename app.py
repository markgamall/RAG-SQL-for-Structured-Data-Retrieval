from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any
import os

from query_chain import create_query_chain


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the query chain (this will load models - might take a moment)
logger.info("Initializing Query-to-SQL chain...")
try:
    # You can customize the persist directory
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    query_chain = create_query_chain(top_k=2, persist_directory=persist_dir)
    logger.info("Query-to-SQL chain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize query chain: {e}")
    query_chain = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Query-to-SQL API is running",
        "chain_status": "initialized" if query_chain else "failed"
    }), 200


@app.route('/schema/list', methods=['GET'])
def list_schema_chunks():
    """List all stored schema chunks"""
    try:
        if query_chain is None:
            return jsonify({
                "error": "Query chain not initialized"
            }), 500
        
        chunks = query_chain.rag_retriever.list_stored_chunks()
        return jsonify({
            "schema_chunks": chunks,
            "count": len(chunks)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing schema chunks: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/schema/update', methods=['POST'])
def update_schema_chunk():
    """Update or add a schema chunk"""
    try:
        if query_chain is None:
            return jsonify({
                "error": "Query chain not initialized"
            }), 500
        
        if not request.is_json:
            return jsonify({
                "error": "Invalid request format",
                "message": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        
        if 'chunk_id' not in data or 'content' not in data:
            return jsonify({
                "error": "Missing parameters",
                "message": "Request body must contain 'chunk_id' and 'content' fields"
            }), 400
        
        chunk_id = data['chunk_id']
        content = data['content']
        
        query_chain.rag_retriever.update_schema_chunk(chunk_id, content)
        
        return jsonify({
            "message": f"Schema chunk '{chunk_id}' updated successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating schema chunk: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/schema/clear', methods=['POST'])
def clear_schema():
    """Clear all schema chunks (use with caution!)"""
    try:
        if query_chain is None:
            return jsonify({
                "error": "Query chain not initialized"
            }), 500
        
        query_chain.rag_retriever.clear_collection()
        
        return jsonify({
            "message": "All schema chunks cleared successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing schema: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/query-to-sql', methods=['POST'])
def query_to_sql():
    """
    Main endpoint to convert natural language query to SQL
    
    Request body: {"query": "your natural language query"}
    Response: {"sql_query": "generated SQL query"}
    """
    try:
        # Check if chain is initialized
        if query_chain is None:
            return jsonify({
                "error": "Query chain not initialized",
                "message": "The system is not ready to process queries"
            }), 500
        
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Invalid request format",
                "message": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        
        # Validate query parameter
        if 'query' not in data:
            return jsonify({
                "error": "Missing query parameter",
                "message": "Request body must contain 'query' field"
            }), 400
        
        user_query = data['query'].strip()
        
        if not user_query:
            return jsonify({
                "error": "Empty query",
                "message": "Query cannot be empty"
            }), 400
        
        logger.info(f"Received query: {user_query}")
        
        # Process the query
        sql_query = query_chain.get_sql_only(user_query)
        
        # Check if there was an error in processing
        if sql_query.startswith("Error:"):
            return jsonify({
                "error": "Processing failed",
                "message": sql_query
            }), 500
        
        # Return successful response
        response = {
            "sql_query": sql_query
        }
        
        logger.info(f"Generated SQL: {sql_query}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in query_to_sql: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your query"
        }), 500


@app.route('/query-to-sql/detailed', methods=['POST'])
def query_to_sql_detailed():
    """
    Endpoint to get detailed response with all intermediate steps
    
    Request body: {"query": "your natural language query"}
    Response: Detailed response with reasoning, schema context, etc.
    """
    try:
        # Check if chain is initialized
        if query_chain is None:
            return jsonify({
                "error": "Query chain not initialized",
                "message": "The system is not ready to process queries"
            }), 500
        
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Invalid request format",
                "message": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        
        # Validate query parameter
        if 'query' not in data:
            return jsonify({
                "error": "Missing query parameter",
                "message": "Request body must contain 'query' field"
            }), 400
        
        user_query = data['query'].strip()
        
        if not user_query:
            return jsonify({
                "error": "Empty query",
                "message": "Query cannot be empty"
            }), 400
        
        logger.info(f"Received detailed query request: {user_query}")
        
        # Process the query with detailed response
        detailed_response = query_chain.get_detailed_response(user_query)
        
        logger.info("Generated detailed response")
        
        return jsonify(detailed_response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in query_to_sql_detailed: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your query"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    # Run the Flask app
    logger.info("Starting Flask API server...")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )