import os
import logging
from typing import List, Dict, Any, Optional, Callable
from functools import partial

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain.docstore.document import Document

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Advanced document processing with optimized chunking strategies
    and multi-format support.
    """
    
    def __init__(self, 
                 chunk_size: int = config.CHUNK_SIZE,
                 chunk_overlap: int = config.CHUNK_OVERLAP,
                 semantic_chunking: bool = config.SEMANTIC_CHUNKING):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of chunks when splitting documents
            chunk_overlap: Overlap between chunks
            semantic_chunking: Whether to use semantic chunking instead of fixed size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_chunking = semantic_chunking
        
        # Define file loaders
        self.file_loaders = {
            ".txt": partial(TextLoader, encoding="utf-8"),
            ".md": UnstructuredMarkdownLoader,
            ".pdf": PyPDFLoader,
            ".csv": partial(CSVLoader, encoding="utf-8"),
        }
        
        # Define text splitters for different document types
        self.text_splitters = {
            "default": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            "python": PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        }
    
    def _get_file_loader(self, file_path: str) -> Callable:
        """Get the appropriate file loader for a given file path"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in self.file_loaders:
            return self.file_loaders[file_extension]
        else:
            # Default to text loader for unknown file types
            logger.warning(f"Unknown file type: {file_extension}. Using text loader as default.")
            return partial(TextLoader, encoding="utf-8")
    
    def _get_text_splitter(self, file_path: str):
        """Get the appropriate text splitter for a given file path"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".md":
            return self.text_splitters["markdown"]
        elif file_extension in [".py", ".js", ".java", ".cpp", ".c"]:
            return self.text_splitters["python"]
        else:
            return self.text_splitters["default"]
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from a file
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading document: {file_path}")
        
        # Get the appropriate loader for the file type
        loader_class = self._get_file_loader(file_path)
        
        try:
            # Load the document
            loader = loader_class(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def _apply_semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Apply semantic chunking to documents
        This is a more advanced chunking strategy that tries to preserve
        semantic meaning rather than just splitting on character count.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        # This is a simplified implementation of semantic chunking
        # In a real-world scenario, you might use more advanced techniques,
        # such as splitting based on sentence embeddings similarity
        
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Split by paragraphs first
            paragraphs = content.split("\n\n")
            
            current_chunk = ""
            current_size = 0
            
            for paragraph in paragraphs:
                paragraph_size = len(paragraph)
                
                # If adding this paragraph would exceed chunk size,
                # save the current chunk and start a new one
                if current_size + paragraph_size > self.chunk_size and current_size > 0:
                    chunked_docs.append(Document(page_content=current_chunk, metadata=metadata.copy()))
                    
                    # Start new chunk with overlap
                    overlap_size = min(self.chunk_overlap, current_size)
                    if overlap_size > 0:
                        # Get the last part of the current chunk for overlap
                        overlap_text = current_chunk[-overlap_size:]
                        current_chunk = overlap_text + "\n\n"
                        current_size = len(current_chunk)
                    else:
                        current_chunk = ""
                        current_size = 0
                
                # Add paragraph to current chunk
                if current_size > 0:
                    current_chunk += "\n\n" + paragraph
                    current_size += paragraph_size + 2  # +2 for "\n\n"
                else:
                    current_chunk = paragraph
                    current_size = paragraph_size
            
            # Add the last chunk if not empty
            if current_size > 0:
                chunked_docs.append(Document(page_content=current_chunk, metadata=metadata.copy()))
        
        logger.info(f"Applied semantic chunking. Original docs: {len(documents)}, Chunked docs: {len(chunked_docs)}")
        return chunked_docs
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a document from a file, including loading and chunking
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed Document objects
        """
        # Load the document
        documents = self.load_document(file_path)
        
        # Apply chunking
        if self.semantic_chunking:
            return self._apply_semantic_chunking(documents)
        else:
            # Get the appropriate text splitter
            text_splitter = self._get_text_splitter(file_path)
            
            # Split the documents
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Applied standard chunking. Original docs: {len(documents)}, Chunked docs: {len(chunked_docs)}")
            
            return chunked_docs
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed Document objects
        """
        all_docs = []
        
        for file_path in file_paths:
            docs = self.process_document(file_path)
            all_docs.extend(docs)
        
        logger.info(f"Processed {len(file_paths)} files into {len(all_docs)} document chunks")
        return all_docs
    
    def process_directory(self, directory_path: str, 
                         include_extensions: Optional[List[str]] = None,
                         exclude_extensions: Optional[List[str]] = None) -> List[Document]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to the directory to process
            include_extensions: List of file extensions to include (e.g., [".txt", ".pdf"])
            exclude_extensions: List of file extensions to exclude
            
        Returns:
            List of processed Document objects
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Not a directory: {directory_path}")
        
        file_paths = []
        
        # Walk through the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                # Check if the file should be included
                if include_extensions and file_extension not in include_extensions:
                    continue
                
                # Check if the file should be excluded
                if exclude_extensions and file_extension in exclude_extensions:
                    continue
                
                # Check if we have a loader for this file type
                if file_extension in self.file_loaders:
                    file_paths.append(file_path)
        
        logger.info(f"Found {len(file_paths)} supported files in {directory_path}")
        
        # Process all found files
        return self.process_documents(file_paths) 