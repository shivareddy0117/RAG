import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

from langchain.retrievers.base import BaseRetriever
from langchain.schema import Document, BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.base import VectorStore

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Class to store retrieval results with scores"""
    documents: List[Document]
    scores: List[float]
    
    def __post_init__(self):
        """Validate that documents and scores have the same length"""
        if len(self.documents) != len(self.scores):
            raise ValueError(
                f"Number of documents ({len(self.documents)}) does not match "
                f"number of scores ({len(self.scores)})"
            )

class HybridRetriever(BaseRetriever):
    """
    Advanced hybrid retriever that combines dense retrieval (embeddings) 
    with sparse retrieval (BM25) for improved performance.
    """
    
    def __init__(self, 
                vectorstore: VectorStore,
                sparse_retriever: Optional[BaseRetriever] = None,
                dense_weight: float = 0.5,
                k: int = config.SIMILARITY_TOP_K,
                score_threshold: float = config.SIMILARITY_SCORE_THRESHOLD):
        """
        Initialize the hybrid retriever
        
        Args:
            vectorstore: Vector store for dense retrieval
            sparse_retriever: Optional sparse retriever (BM25, tf-idf, etc.)
            dense_weight: Weight to apply to dense retrieval scores (0-1)
            k: Number of documents to retrieve
            score_threshold: Minimum score threshold for including results
        """
        self.vectorstore = vectorstore
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.k = k
        self.score_threshold = score_threshold
        
        # Validate parameters
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using a hybrid approach
        
        Args:
            query: Query to retrieve documents for
            
        Returns:
            List of relevant Document objects
        """
        hybrid_results = self.hybrid_search(query, self.k)
        return hybrid_results.documents
    
    def _dense_search(self, query: str, k: int) -> RetrievalResult:
        """
        Perform dense retrieval using vector similarity search
        
        Args:
            query: Query to search for
            k: Number of results to return
            
        Returns:
            RetrievalResult with documents and scores
        """
        logger.info(f"Performing dense retrieval for query: {query}")
        
        try:
            if hasattr(self.vectorstore, "similarity_search_with_score"):
                # Get documents with scores
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                
                # Split into documents and scores
                documents = [doc for doc, _ in results]
                scores = [score for _, score in results]
                
                # Normalize scores to 0-1 range
                if scores:
                    min_score = min(scores)
                    max_score = max(scores)
                    if min_score != max_score:
                        scores = [(s - min_score) / (max_score - min_score) for s in scores]
                    else:
                        scores = [1.0] * len(scores)
                
                return RetrievalResult(documents=documents, scores=scores)
            else:
                # Fallback for vector stores that don't support scores
                documents = self.vectorstore.similarity_search(query, k=k)
                scores = [1.0] * len(documents)  # Default score
                return RetrievalResult(documents=documents, scores=scores)
        except Exception as e:
            logger.error(f"Error during dense retrieval: {str(e)}")
            return RetrievalResult(documents=[], scores=[])
    
    def _sparse_search(self, query: str, k: int) -> RetrievalResult:
        """
        Perform sparse retrieval (keyword-based search)
        
        Args:
            query: Query to search for
            k: Number of results to return
            
        Returns:
            RetrievalResult with documents and scores
        """
        if self.sparse_retriever is None:
            return RetrievalResult(documents=[], scores=[])
        
        logger.info(f"Performing sparse retrieval for query: {query}")
        
        try:
            # Get documents from sparse retriever
            # Note: This assumes the sparse retriever has a method to return scores
            # In a real implementation, you might need to handle different retriever types
            
            # For now, we'll assume it doesn't return scores and just assign
            # default scores
            documents = self.sparse_retriever.get_relevant_documents(query)
            scores = [1.0] * len(documents)  # Default score
            
            return RetrievalResult(documents=documents, scores=scores)
        except Exception as e:
            logger.error(f"Error during sparse retrieval: {str(e)}")
            return RetrievalResult(documents=[], scores=[])
    
    def _merge_results(self, 
                      dense_results: RetrievalResult, 
                      sparse_results: RetrievalResult,
                      k: int) -> RetrievalResult:
        """
        Merge results from dense and sparse retrievers
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: Number of results to return
            
        Returns:
            Merged results
        """
        # Create a dictionary to track document scores by content hash
        merged_docs = {}
        
        # Process dense results
        for doc, score in zip(dense_results.documents, dense_results.scores):
            doc_hash = hash(doc.page_content)
            
            if doc_hash not in merged_docs:
                merged_docs[doc_hash] = {
                    "doc": doc,
                    "dense_score": score * self.dense_weight,
                    "sparse_score": 0,
                    "combined_score": score * self.dense_weight
                }
        
        # Process sparse results and merge with dense results
        for doc, score in zip(sparse_results.documents, sparse_results.scores):
            doc_hash = hash(doc.page_content)
            
            if doc_hash in merged_docs:
                # Update existing document
                merged_docs[doc_hash]["sparse_score"] = score * self.sparse_weight
                merged_docs[doc_hash]["combined_score"] += score * self.sparse_weight
            else:
                # Add new document
                merged_docs[doc_hash] = {
                    "doc": doc,
                    "dense_score": 0,
                    "sparse_score": score * self.sparse_weight,
                    "combined_score": score * self.sparse_weight
                }
        
        # Sort by combined score and filter by threshold
        sorted_results = sorted(
            merged_docs.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        # Filter by score threshold
        filtered_results = [
            r for r in sorted_results 
            if r["combined_score"] >= self.score_threshold
        ]
        
        # Take top k
        top_k_results = filtered_results[:k]
        
        # Extract documents and scores
        documents = [r["doc"] for r in top_k_results]
        scores = [r["combined_score"] for r in top_k_results]
        
        logger.info(f"Merged results: {len(documents)} documents")
        
        return RetrievalResult(documents=documents, scores=scores)
    
    def hybrid_search(self, query: str, k: int) -> RetrievalResult:
        """
        Perform hybrid search combining dense and sparse retrievals
        
        Args:
            query: Query to search for
            k: Number of results to return
            
        Returns:
            RetrievalResult with documents and scores
        """
        # Perform dense retrieval
        dense_results = self._dense_search(query, k=k)
        
        # If dense_weight is 1.0, we can skip sparse retrieval
        if self.dense_weight == 1.0 or self.sparse_retriever is None:
            return dense_results
        
        # If dense_weight is 0, we can skip dense retrieval
        if self.dense_weight == 0.0:
            return self._sparse_search(query, k=k)
        
        # Perform sparse retrieval
        sparse_results = self._sparse_search(query, k=k)
        
        # Merge results
        return self._merge_results(dense_results, sparse_results, k=k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query
        
        Args:
            query: Query to retrieve documents for
            
        Returns:
            List of relevant Document objects
        """
        return self._get_relevant_documents(query) 