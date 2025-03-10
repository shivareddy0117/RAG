import os
import argparse
import logging
from typing import List, Dict, Any, Optional

from src.document_loaders.document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from src.evaluation.rag_evaluator import RAGEvaluator
from src.utils.gpu_utils import setup_gpu, clean_gpu_memory
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced RAG System with optimized retrieval and GPU acceleration"
    )
    
    # Data processing arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=config.DATA_DIR,
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--file_types", 
        type=str, 
        default=".txt,.pdf,.md,.csv",
        help="Comma-separated list of file extensions to process"
    )
    
    # Vector store arguments
    parser.add_argument(
        "--vectorstore_path", 
        type=str, 
        default=config.VECTORSTORE_PATH,
        help="Path to save/load the vector store"
    )
    parser.add_argument(
        "--rebuild_vectorstore", 
        action="store_true",
        help="Force rebuilding the vector store even if it exists"
    )
    
    # Model arguments
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        default=config.EMBEDDING_MODEL,
        help="Embedding model to use"
    )
    parser.add_argument(
        "--llm_model", 
        type=str, 
        default=config.LLM_MODEL,
        help="LLM model to use"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--use_gpu", 
        action="store_true", 
        default=config.USE_GPU,
        help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true", 
        default=config.MIXED_PRECISION,
        help="Use mixed precision for faster inference"
    )
    
    # Mode arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["interactive", "query", "evaluate", "benchmark"],
        default="interactive",
        help="Mode to run the application in"
    )
    parser.add_argument(
        "--query", 
        type=str,
        help="Query to run in query mode"
    )
    parser.add_argument(
        "--evaluation_file", 
        type=str,
        help="File containing evaluation queries (one per line)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="output/results.json",
        help="Path to save evaluation results"
    )
    
    return parser.parse_args()

def load_or_create_rag_pipeline(args):
    """
    Load or create the RAG pipeline based on command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Initialized RAG pipeline
    """
    # Set up GPU if requested
    gpu_info = setup_gpu(
        use_gpu=args.use_gpu,
        mixed_precision=args.mixed_precision
    )
    logger.info(f"GPU setup: {gpu_info}")
    
    vectorstore_path = args.vectorstore_path
    
    # Check if vector store exists and we don't need to rebuild
    if os.path.exists(vectorstore_path) and not args.rebuild_vectorstore:
        logger.info(f"Loading existing vector store from {vectorstore_path}")
        
        # Load existing RAG pipeline
        rag_pipeline = RAGPipeline.load(
            directory=vectorstore_path
        )
        
        logger.info("RAG pipeline loaded successfully")
        return rag_pipeline
    else:
        # Create new RAG pipeline
        logger.info("Creating new RAG pipeline")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            use_gpu=args.use_gpu,
            mixed_precision=args.mixed_precision
        )
        
        # Process documents
        file_types = [ft.strip() for ft in args.file_types.split(",")]
        
        logger.info(f"Processing documents from {args.data_dir} with types {file_types}")
        
        # Initialize document processor
        doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            semantic_chunking=config.SEMANTIC_CHUNKING
        )
        
        # Process documents
        documents = doc_processor.process_directory(
            args.data_dir,
            include_extensions=file_types
        )
        
        if not documents:
            logger.warning(f"No documents found in {args.data_dir} with types {file_types}")
            return None
        
        logger.info(f"Processed {len(documents)} documents")
        
        # Add documents to RAG pipeline
        rag_pipeline.add_documents(documents)
        
        # Build vector store
        rag_pipeline.build_vectorstore()
        
        # Build retriever
        rag_pipeline.build_retriever()
        
        # Build QA chain
        rag_pipeline.build_qa_chain()
        
        # Save the vector store
        os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
        rag_pipeline.save(vectorstore_path)
        
        logger.info(f"RAG pipeline created and saved to {vectorstore_path}")
        return rag_pipeline

def interactive_mode(rag_pipeline):
    """
    Run the application in interactive mode
    
    Args:
        rag_pipeline: Initialized RAG pipeline
    """
    logger.info("Starting interactive mode. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter your question (or 'exit' to quit): ")
            
            if query.lower() == "exit":
                break
            
            if not query.strip():
                continue
            
            # Process the query
            result = rag_pipeline.query(query)
            
            # Print the answer
            print("\nAnswer:")
            print(result["answer"])
            
            # Print sources
            if "source_documents" in result and result["source_documents"]:
                print("\nSources:")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"\n[{i+1}] {doc.metadata.get('source', 'Unknown source')}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")
    
    logger.info("Interactive mode ended")

def query_mode(rag_pipeline, query):
    """
    Run the application in query mode
    
    Args:
        rag_pipeline: Initialized RAG pipeline
        query: Query to run
    """
    logger.info(f"Running query: {query}")
    
    try:
        # Process the query
        result = rag_pipeline.query(query)
        
        # Print the answer
        print("\nAnswer:")
        print(result["answer"])
        
        # Print sources
        if "source_documents" in result and result["source_documents"]:
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"\n[{i+1}] {doc.metadata.get('source', 'Unknown source')}")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        print(f"Error: {str(e)}")

def evaluation_mode(rag_pipeline, evaluation_file, output_file):
    """
    Run the application in evaluation mode
    
    Args:
        rag_pipeline: Initialized RAG pipeline
        evaluation_file: File containing evaluation queries
        output_file: Path to save evaluation results
    """
    if not os.path.exists(evaluation_file):
        logger.error(f"Evaluation file not found: {evaluation_file}")
        return
    
    logger.info(f"Running evaluation with queries from {evaluation_file}")
    
    # Read evaluation queries
    with open(evaluation_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(queries)} evaluation queries")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_pipeline)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(queries)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    evaluator.save_results(output_file)
    
    # Print summary
    summary = evaluator.summarize_results()
    
    print("\nEvaluation Summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value}")
    
    logger.info(f"Evaluation completed. Results saved to {output_file}")

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Load or create RAG pipeline
    rag_pipeline = load_or_create_rag_pipeline(args)
    
    if rag_pipeline is None:
        logger.error("Failed to initialize RAG pipeline")
        return
    
    # Run in the specified mode
    try:
        if args.mode == "interactive":
            interactive_mode(rag_pipeline)
        elif args.mode == "query":
            if not args.query:
                logger.error("Query not provided for query mode")
                return
            query_mode(rag_pipeline, args.query)
        elif args.mode == "evaluate":
            if not args.evaluation_file:
                logger.error("Evaluation file not provided for evaluation mode")
                return
            evaluation_mode(rag_pipeline, args.evaluation_file, args.output_file)
        elif args.mode == "benchmark":
            # Not implemented yet
            logger.error("Benchmark mode not implemented yet")
            return
    finally:
        # Clean up GPU memory
        if args.use_gpu:
            clean_gpu_memory()

if __name__ == "__main__":
    main() 