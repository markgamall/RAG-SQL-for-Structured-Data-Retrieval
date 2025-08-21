import chromadb
import google.generativeai as genai
from typing import List, Tuple
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class RAGRetriever:
    def __init__(self, threshold: float = 1.75, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "schema_chunks"
        self.threshold = threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configure Google API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self._initialize_collection()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using Google's embedding model"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return []
    
    def _initialize_collection(self):
        """Initialize the ChromaDB collection with schema chunks"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            
            # Check if collection has data
            count = self.collection.count()
            if count > 0:
                self.logger.info(f"Found existing collection with {count} documents")
            else:
                self.logger.info("Collection exists but is empty, populating...")
                self._populate_collection()
                
        except Exception as e:
            # Create new collection if it doesn't exist
            self.logger.info(f"Creating new collection: {e}")
            self.collection = self.client.create_collection(name=self.collection_name)
            self._populate_collection()
    
    def _populate_collection(self):
        """Populate the collection with schema chunks"""
        self.logger.info("Populating collection with schema chunks and generating embeddings...")
        
        schema_chunks = [
            {
                "id": "july_HCPs",
                "content": """TABLE: july_HCPs
    Database: healthcare_analytics
    Description: Stores information about healthcare professionals (HCPs) for July data, including personal details, roles, and classifications.
    Columns:
    - id (VARCHAR(255), PRIMARY KEY): Unique identifier for the HCP.
    - customerid (INT, UNIQUE): Unique customer reference ID.
    - englishname (TEXT): Full name of the HCP in English.
    - isconsultant (BOOLEAN): Whether the HCP is a consultant (0 for false, 1 for true).
    - isdecisionmaker (BOOLEAN): Whether the HCP is a decision-maker (0 for false, 1 for true).
    - issamspeaker (BOOLEAN): Whether the HCP is a SAM speaker (0 for false, 1 for true).
    - isuniversitystaff (BOOLEAN): Whether the HCP is part of university staff (0 for false, 1 for true).
    - isampmspeaker (BOOLEAN): Whether the HCP is an AM/PM speaker (0 for false, 1 for true).
    - customerclassificationid (INT): ID representing the HCP's classification.
    - CustomerClassification (TEXT): Description of the classification.
    - specialityid (INT): ID of the HCP's specialty.
    - Speciality (TEXT): Name of the specialty.
    - countryid (INT): ID of the country where the HCP is located.
    - Country (VARCHAR(255)): Name of the country.
    Character Set: utf8mb4 with utf8mb4_unicode_ci collation
    Relationships:
    - Referenced by `july_interactions.HCPId` (foreign key) — links a medical representative's interaction to a specific HCP.
    
    Common Query Patterns:
    - Filter by country: WHERE Country = 'Egypt' or WHERE LOWER(Country) LIKE '%egypt%'
    - Filter by specialty: WHERE Speciality = 'Cardiology' or similar
    - Filter by boolean flags: WHERE isconsultant = 1 (for true) or WHERE isconsultant = 0 (for false)
    - Customer lookup: WHERE customerid = [number] or WHERE id = '[id_string]'"""
            },
            {
                "id": "july_interactions",
                "content": """TABLE: july_interactions
    Database: healthcare_analytics
    Description: Records medical representatives' interactions with healthcare professionals for July data, including meeting details, status, and business line information.
    Columns:
    - MRId (VARCHAR(255)): Identifier for the medical representative.
    - MRArFullName (TEXT): Full name of the medical representative in Arabic.
    - InteractionId (INT, PRIMARY KEY): Unique identifier for the interaction.
    - InteractionStatusId (INT): Numeric status code of the interaction.
    - InteractionStatus (VARCHAR(255)): Description of the interaction status.
    - reportdate (DATETIME): Date and time of the interaction report.
    - lineid (INT): ID of the medical line involved in the interaction.
    - LineName (VARCHAR(255)): Name of the medical line.
    - businessUnitId (INT): ID of the business unit.
    - BusinessUnitName (VARCHAR(255)): Name of the business unit.
    - HCPId (VARCHAR(255)): Foreign key referencing the `july_HCPs.id` column.
    - HCPCustomerId (INT): Customer ID of the HCP involved.
    - HCPEnglishName (TEXT): English name of the HCP involved.
    - HCPArabicName (TEXT): Arabic name of the HCP involved.
    - SpecialtyId (INT): ID of the HCP's specialty.
    - Specialty (TEXT): Name of the HCP's specialty.
    Character Set: utf8mb4 with utf8mb4_unicode_ci collation
    Relationships:
    - Foreign key (`HCPId`) references `july_HCPs.id` — associates a medical representative with a specific healthcare professional.
    
    Common Query Patterns:
    - Join with HCPs: JOIN july_HCPs ON july_interactions.HCPId = july_HCPs.id
    - Filter by date: WHERE DATE(reportdate) = '2024-07-15' or WHERE reportdate BETWEEN '2024-07-01' AND '2024-07-31'
    - Filter by status: WHERE InteractionStatus = 'Completed' or WHERE InteractionStatusId = [number]
    - Filter by business unit: WHERE BusinessUnitName = '[unit_name]' or WHERE businessUnitId = [number]
    - Aggregate by MR: GROUP BY MRId, MRArFullName"""
            },
            {
                "id": "database_context",
                "content": """DATABASE: healthcare_analytics
    Description: Healthcare analytics database containing July 2024 data for healthcare professionals and medical representative interactions.
    
    Key Information:
    - Database name: healthcare_analytics
    - Main tables: july_HCPs, july_interactions
    - Character encoding: utf8mb4 with utf8mb4_unicode_ci collation
    - Date format in july_interactions: DATETIME format (YYYY-MM-DD HH:MM:SS)
    - Boolean fields in july_HCPs: Stored as TINYINT (0 for false, 1 for true)
    - Primary data source: CSV files loaded via LOAD DATA INFILE
    
    Sample Alternative Databases Available:
    - employees: Sample employee database
    - sakila: Sample DVD rental database
    
    Important Notes:
    - Always use the correct table names: july_HCPs and july_interactions (not HCP or MedicalReps)
    - Country names should be compared with proper case sensitivity
    - Text fields use TEXT data type which allows for longer content
    - Foreign key relationship exists between july_interactions.HCPId and july_HCPs.id"""
            }
        ]
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for chunk in schema_chunks:
            self.logger.info(f"Generating embedding for {chunk['id']}...")
            embedding = self._get_embedding(chunk["content"])
            if embedding:  # Only add if embedding was successful
                documents.append(chunk["content"])
                embeddings.append(embedding)
                ids.append(chunk["id"])
                metadatas.append({"table": chunk["id"]})
            else:
                self.logger.warning(f"Failed to generate embedding for {chunk['id']}")
        
        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            self.logger.info(f"Successfully added {len(documents)} documents with embeddings to collection")
        else:
            self.logger.error("No embeddings were generated successfully")
    
    def update_schema_chunk(self, chunk_id: str, content: str):
        """
        Update or add a specific schema chunk
        
        Args:
            chunk_id (str): Unique identifier for the chunk
            content (str): Schema content
        """
        self.logger.info(f"Updating schema chunk: {chunk_id}")
        
        embedding = self._get_embedding(content)
        if embedding:
            # ChromaDB upsert - will update if exists, create if not
            self.collection.upsert(
                ids=[chunk_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"table": chunk_id}]
            )
            self.logger.info(f"Successfully updated chunk: {chunk_id}")
        else:
            self.logger.error(f"Failed to generate embedding for chunk: {chunk_id}")
    
    def delete_schema_chunk(self, chunk_id: str):
        """
        Delete a specific schema chunk
        
        Args:
            chunk_id (str): Unique identifier for the chunk to delete
        """
        try:
            self.collection.delete(ids=[chunk_id])
            self.logger.info(f"Successfully deleted chunk: {chunk_id}")
        except Exception as e:
            self.logger.error(f"Error deleting chunk {chunk_id}: {e}")
    
    def list_stored_chunks(self) -> List[str]:
        """
        Get list of all stored chunk IDs
        
        Returns:
            List[str]: List of chunk IDs
        """
        try:
            results = self.collection.get()
            return results["ids"]
        except Exception as e:
            self.logger.error(f"Error listing chunks: {e}")
            return []
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.logger.info("Collection cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
    
    def retrieve_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top k similar schema chunks based on the query
        
        Args:
            query (str): User query
            k (int): Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples
        """
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        docs = results["documents"][0]
        scores = results["distances"][0]
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in zip(docs, scores) if score <= self.threshold]
        
        return filtered
    
    def get_schema_context(self, query: str, k: int = 3) -> str:
        """
        Get formatted schema context for LLM consumption
        
        Args:
            query (str): User query
            k (int): Number of top chunks to retrieve
            
        Returns:
            str: Formatted schema context
        """
        chunks = self.retrieve_chunks(query, k)
        
        if not chunks:
            return "No relevant schema found."
        
        context = "Relevant Database Schema:\n\n"
        for i, (doc, score) in enumerate(chunks, 1):
            context += f"Schema {i} (Relevance Score: {score:.3f}):\n{doc}\n\n"
        
        return context
