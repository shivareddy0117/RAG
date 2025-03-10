import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union

import config
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Advanced Retrieval-Augmented Generation Pipeline implementing hybrid search,
    optimized context processing, and GPU acceleration.
    """
    
    def __init__(self, 
                 documents: Optional[List[Any]] = None,
                 use_gpu: bool = config.USE_GPU,
                 mixed_precision: bool = config.MIXED_PRECISION):
        """
        Initialize the RAG pipeline with documents and configuration
        
        Args:
            documents: Optional pre-loaded documents
            use_gpu: Whether to use GPU acceleration
            mixed_precision: Whether to use mixed precision for inference
        """
        self.documents = documents
        
        # Configure GPU settings
        if use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Set up mixed precision if enabled
            if mixed_precision:
                logger.info("Enabling mixed precision training")
                torch.set_float32_matmul_precision('high')
        else:
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but not available. Using CPU instead.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Initialize vector store and retriever
        self.vectorstore = None
        self.retriever = None
        
        # Initialize LLM
        self._init_llm()
        
        # Initialize QA chain
        self.qa_chain = None
    
    def _init_embeddings(self):
        """Initialize the embedding model"""
        logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL}")
        
        # Set device for embeddings
        device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        
        # Use HuggingFace embeddings with optimized settings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": config.EMBEDDING_BATCH_SIZE,
                "normalize_embeddings": True,
                "device": device
            }
        )
        
        logger.info(f"Embeddings initialized on {device}")
    
    def _init_llm(self):
        """Initialize the language model"""
        logger.info(f"Initializing LLM: {config.LLM_MODEL}")
        
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.OPENAI_API_KEY
        )
    
    def add_documents(self, documents: List[Any]):
        """
        Add documents to the pipeline
        
        Args:
            documents: List of documents to add
        """
        self.documents = documents if self.documents is None else self.documents + documents
        logger.info(f"Added documents. Total documents: {len(self.documents)}")
    
    def build_vectorstore(self, documents: Optional[List[Any]] = None):
        """
        Build the vector store from documents
        
        Args:
            documents: Optional documents to use instead of stored ones
        """
        docs_to_use = documents if documents is not None else self.documents
        
        if docs_to_use is None or len(docs_to_use) == 0:
            raise ValueError("No documents provided for building the vector store")
        
        logger.info(f"Building vector store with {len(docs_to_use)} documents")
        
        if config.VECTORDB_TYPE == "FAISS":
            self.vectorstore = FAISS.from_documents(docs_to_use, self.embeddings)
            logger.info("FAISS vector store created successfully")
        elif config.VECTORDB_TYPE == "Chroma":
            self.vectorstore = Chroma.from_documents(docs_to_use, self.embeddings)
            logger.info("Chroma vector store created successfully")
        else:
            raise ValueError(f"Unsupported vector store type: {config.VECTORDB_TYPE}")
    
    def build_retriever(self):
        """Build the retriever with advanced configuration"""
        if self.vectorstore is None:
            raise ValueError("Vector store must be built before creating a retriever")
        
        # Create base retriever
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": config.SIMILARITY_TOP_K,
                "score_threshold": config.SIMILARITY_SCORE_THRESHOLD
            }
        )
        
        # Apply context compression if enabled
        if config.RERANKING_ENABLED:
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            logger.info("Created retriever with contextual compression")
        else:
            self.retriever = base_retriever
            logger.info("Created base retriever")
    
    def build_qa_chain(self):
        """Build the QA chain with optimized prompt template"""
        if self.retriever is None:
            raise ValueError("Retriever must be built before creating a QA chain")
        
        # Define prompt template with nuanced instructions for better responses
        template = """
        You are an AI assistant providing accurate and helpful information.
        
        Context information is below:
        ---------------------
        {context}
        ---------------------
        
        Given this context, please answer the question: {question}
        
        If the context doesn't contain the information needed to answer the question,
        just say "I don't have enough information to answer that question." Don't try
        to make up an answer.
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("QA chain built successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain is not initialized. Call build_qa_chain first.")
        
        logger.info(f"Processing query: {question}")
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def save(self, directory: str = config.VECTORSTORE_PATH):
        """
        Save the vector store to disk
        
        Args:
            directory: Directory to save to
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized")
        
        os.makedirs(directory, exist_ok=True)
        
        if isinstance(self.vectorstore, FAISS):
            self.vectorstore.save_local(directory)
            logger.info(f"FAISS vector store saved to {directory}")
        elif isinstance(self.vectorstore, Chroma):
            # Chroma has its own persistence mechanism
            logger.info("Chroma vector store is already persistent")
        
    @classmethod
    def load(cls, 
             directory: str = config.VECTORSTORE_PATH, 
             embedding_model: Optional[Any] = None):
        """
        Load a saved pipeline from disk
        
        Args:
            directory: Directory to load from
            embedding_model: Optional embedding model to use
            
        Returns:
            Loaded RAG pipeline
        """
        instance = cls()
        
        if embedding_model is not None:
            instance.embeddings = embedding_model
        
        if os.path.exists(directory):
            logger.info(f"Loading vector store from {directory}")
            
            if config.VECTORDB_TYPE == "FAISS":
                instance.vectorstore = FAISS.load_local(
                    directory,
                    instance.embeddings
                )
                logger.info("FAISS vector store loaded successfully")
            else:
                raise ValueError(f"Loading from disk not implemented for {config.VECTORDB_TYPE}")
                
            # Initialize other components
            instance.build_retriever()
            instance.build_qa_chain()
            
            return instance
        else:
            raise FileNotFoundError(f"Directory not found: {directory}") 