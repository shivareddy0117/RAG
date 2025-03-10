#!/usr/bin/env python
# coding: utf-8

"""
Advanced RAG System Demo

This script demonstrates the capabilities of our advanced Retrieval-Augmented Generation (RAG) 
system with optimized retrieval techniques and GPU acceleration.
"""

# Import necessary libraries
import os
import sys
import logging
import time
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append('..')

# Load environment variables from .env file
load_dotenv()

# Import our modules
from rag_pipeline import RAGPipeline
from src.document_loaders.document_processor import DocumentProcessor
from src.utils.gpu_utils import setup_gpu, get_gpu_info
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and setup."""
    print("\n## Setup GPU Acceleration")
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"GPU available: {gpu_info['available']}")
    
    if gpu_info['available']:
        print(f"GPU device: {gpu_info['current_device_name']}")
        print(f"Memory allocated: {gpu_info['memory_allocated']:.2f} MB")
    
    # Setup GPU with mixed precision
    setup_info = setup_gpu(use_gpu=config.USE_GPU, mixed_precision=config.MIXED_PRECISION)
    print(f"\nGPU setup: {setup_info}")

def process_documents():
    """Process documents using advanced chunking techniques."""
    print("\n## Process Documents")
    
    # Initialize the document processor with semantic chunking
    doc_processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        semantic_chunking=config.SEMANTIC_CHUNKING
    )
    
    # Process sample documents
    data_dir = os.path.join('..', config.DATA_DIR)
    documents = doc_processor.process_directory(
        data_dir,
        include_extensions=['.txt', '.md', '.pdf', '.csv']
    )
    
    print(f"Processed {len(documents)} document chunks")
    
    # Display a sample document chunk
    if documents:
        sample_doc = documents[0]
        print(f"\nSample document chunk:")
        print(f"Content (first 200 chars): {sample_doc.page_content[:200]}...")
        print(f"Metadata: {sample_doc.metadata}")
    
    return documents

def build_rag_pipeline(documents):
    """Build the RAG pipeline with the processed documents."""
    print("\n## Build the RAG Pipeline")
    
    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline(
        documents=None,  # We'll add documents after initialization
        use_gpu=config.USE_GPU,
        mixed_precision=config.MIXED_PRECISION
    )
    
    # Add the processed documents
    rag_pipeline.add_documents(documents)
    
    # Build the vector store
    print("Building vector store...")
    rag_pipeline.build_vectorstore()
    
    # Build the retriever with advanced configuration
    print("Building retriever...")
    rag_pipeline.build_retriever()
    
    # Build the QA chain
    print("Building QA chain...")
    rag_pipeline.build_qa_chain()
    
    return rag_pipeline

def run_query(rag_pipeline, query):
    """Run a query and display results."""
    print(f"\nQuery: {query}")
    
    # Time the query processing
    start_time = time.time()
    
    # Process the query
    result = rag_pipeline.query(query)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Display the answer
    print(f"\nAnswer:\n{result['answer']}")
    
    # Display source documents
    print(f"\nSource Documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\n[{i+1}] Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content (excerpt): {doc.page_content[:100]}...")
    
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    return result

def test_rag_system(rag_pipeline):
    """Test the RAG system with sample queries."""
    print("\n## Test the RAG System")
    
    # Test with a simple query
    simple_query = "What are Large Language Models?"
    print("\nRunning simple query...")
    simple_result = run_query(rag_pipeline, simple_query)
    
    # Test with a more complex query
    complex_query = "Explain how LoRA and QLoRA techniques are used for fine-tuning LLMs and what advantages they provide."
    print("\nRunning complex query...")
    complex_result = run_query(rag_pipeline, complex_query)
    
    # Test with a query about RAG
    rag_query = "What is hybrid search in RAG systems and how does it improve retrieval performance?"
    print("\nRunning RAG-specific query...")
    rag_result = run_query(rag_pipeline, rag_query)

def evaluate_rag_performance(rag_pipeline):
    """Evaluate the retrieval performance of the RAG system."""
    print("\n## Evaluate Retrieval Performance")
    
    # Import our evaluation module
    from src.evaluation.rag_evaluator import RAGEvaluator, EvaluationResult
    
    # Initialize the evaluator
    evaluator = RAGEvaluator(rag_pipeline)
    
    # Load evaluation queries
    eval_file = os.path.join('..', config.DATA_DIR, 'evaluation_queries.txt')
    with open(eval_file, 'r') as f:
        eval_queries = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(eval_queries)} evaluation queries")
    print(f"Sample queries:")
    for i, query in enumerate(eval_queries[:3]):
        print(f"[{i+1}] {query}")
    
    # Run evaluation on a subset of queries (for demonstration purposes)
    subset_queries = eval_queries[:3]  # Use first 3 queries for demonstration
    print(f"\nEvaluating {len(subset_queries)} queries...")
    
    # This might take some time, as it evaluates multiple aspects of each result
    eval_results = evaluator.evaluate_dataset(subset_queries)
    
    # Display summary results
    summary = evaluator.summarize_results()
    print("\nEvaluation Summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

def demonstrate_prompt_engineering(rag_pipeline, documents):
    """Showcase advanced prompt engineering techniques."""
    print("\n## Demonstrate Advanced Prompt Engineering")
    
    # Import our prompt templates
    from src.prompt_templates.advanced_prompts import PromptTemplates
    from langchain.chains import LLMChain
    
    # Create a summarization prompt
    summarization_prompt = PromptTemplates.summarization_prompt()
    print(f"Summarization Prompt Template:")
    print(summarization_prompt.template[:200] + "...")
    
    # Create a fact checking prompt
    fact_checking_prompt = PromptTemplates.fact_checking_prompt()
    print(f"\nFact Checking Prompt Template:")
    print(fact_checking_prompt.template[:200] + "...")
    
    # Demonstrate summarization using our advanced prompt
    summarization_chain = LLMChain(
        llm=rag_pipeline.llm,
        prompt=summarization_prompt
    )
    
    # Use one of our document chunks for summarization
    document_to_summarize = documents[0].page_content
    print("\nSummarizing sample document...")
    summary = summarization_chain.run(context=document_to_summarize)
    
    print(f"\nDocument Summary:\n{summary}")

def save_and_load_pipeline(rag_pipeline):
    """Demonstrate saving and loading the RAG pipeline."""
    print("\n## Save and Load the RAG Pipeline")
    
    # Save the vector store
    vectorstore_path = os.path.join('..', config.VECTORSTORE_PATH)
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    
    print(f"Saving RAG pipeline to {vectorstore_path}...")
    rag_pipeline.save(vectorstore_path)
    
    # Load the RAG pipeline from disk
    print(f"\nLoading RAG pipeline from {vectorstore_path}...")
    loaded_pipeline = RAGPipeline.load(vectorstore_path)
    
    # Test the loaded pipeline
    test_query = "What is semantic chunking?"
    print(f"\nTesting loaded pipeline with query: '{test_query}'")
    result = loaded_pipeline.query(test_query)
    print(f"\nAnswer:\n{result['answer']}")

def main():
    """Main function to run the demo."""
    print("# Advanced RAG System Demo")
    print("\nThis script demonstrates the capabilities of our advanced RAG system.")
    
    # Check GPU availability and setup
    check_gpu()
    
    # Process documents
    documents = process_documents()
    
    if not documents:
        print("No documents processed. Exiting.")
        return
    
    # Build the RAG pipeline
    rag_pipeline = build_rag_pipeline(documents)
    
    # Test the RAG system
    test_rag_system(rag_pipeline)
    
    # Evaluate retrieval performance
    evaluate_rag_performance(rag_pipeline)
    
    # Demonstrate advanced prompt engineering
    demonstrate_prompt_engineering(rag_pipeline, documents)
    
    # Save and load the pipeline
    save_and_load_pipeline(rag_pipeline)
    
    print("\n## Conclusion")
    print("\nThis demo has showcased the key capabilities of our advanced RAG system:")
    print("1. Document Processing: Semantic chunking and multi-format support")
    print("2. Optimized Retrieval: Using vector search with GPU acceleration")
    print("3. Advanced Prompt Engineering: Specialized prompts for different tasks")
    print("4. Evaluation Framework: Comprehensive metrics for RAG performance")
    print("5. Persistence: Saving and loading the pipeline")

if __name__ == "__main__":
    main() 