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
                "id": "HCP",
                "content": """TABLE: HCP
    Description: Stores information about healthcare professionals (HCPs), including personal details, roles, and classifications.
    Columns:
    - id (INT, PRIMARY KEY): Unique identifier for the HCP.
    - customerid (INT, UNIQUE): Unique customer reference ID.
    - englishname (VARCHAR(255)): Full name of the HCP in English.
    - isconsultant (BOOLEAN): Whether the HCP is a consultant.
    - isdecisionmaker (BOOLEAN): Whether the HCP is a decision-maker.
    - issamspeaker (BOOLEAN): Whether the HCP is a SAM speaker.
    - isuniversitystaff (BOOLEAN): Whether the HCP is part of university staff.
    - isampmspeaker (BOOLEAN): Whether the HCP is an AM/PM speaker.
    - customerclassificationid (INT): ID representing the HCP's classification.
    - CustomerClassification (VARCHAR(255)): Description of the classification.
    - specialityid (INT): ID of the HCP's specialty.
    - Speciality (VARCHAR(255)): Name of the specialty.
    - countryid (INT): ID of the country where the HCP is located.
    - Country (VARCHAR(255)): Name of the country.
    Relationships:
    - Referenced by `MedicalReps.HCPId` (foreign key) — links a medical representative's interaction to a specific HCP."""
            },
            {
                "id": "MedicalReps",
                "content": """TABLE: MedicalReps
    Description: Records medical representatives' interactions with healthcare professionals, including meeting details, status, and business line.
    Columns:
    - MRId (INT, PRIMARY KEY): Unique identifier for the medical representative.
    - MRArFullName (VARCHAR(255)): Full name of the medical representative (Arabic).
    - InteractionId (INT): ID of the interaction.
    - InteractionStatusId (INT): Numeric status code of the interaction.
    - InteractionStatus (VARCHAR(255)): Description of the interaction status.
    - reportdate (DATE): Date of the interaction report.
    - lineid (INT): ID of the medical line involved in the interaction.
    - LineName (VARCHAR(255)): Name of the medical line.
    - businessUnitId (INT): ID of the business unit.
    - BusinessUnitName (VARCHAR(255)): Name of the business unit.
    - HCPId (INT): Foreign key referencing the `HCP.id` column.
    - HCPCustomerId (INT): Customer ID of the HCP involved.
    - HCPEnglishName (VARCHAR(255)): English name of the HCP involved.
    - HCPArabicName (VARCHAR(255)): Arabic name of the HCP involved.
    - SpecialtyId (INT): ID of the HCP's specialty.
    - Specialty (VARCHAR(255)): Name of the HCP's specialty.
    Relationships:
    - Foreign key (`HCPId`) references `HCP.id` — associates a medical representative with a specific healthcare professional."""
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
    
    def retrieve_chunks(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
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
    
    def get_schema_context(self, query: str, k: int = 2) -> str:
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