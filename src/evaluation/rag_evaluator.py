import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Class to store evaluation results"""
    question: str
    answer: str
    expected_answer: Optional[str] = None
    retrieved_documents: Optional[List[Document]] = None
    relevance_scores: Optional[List[float]] = None
    faithfulness_score: Optional[float] = None
    answer_relevance_score: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    latency_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary"""
        return {
            "question": self.question,
            "answer": self.answer,
            "expected_answer": self.expected_answer,
            "retrieved_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in (self.retrieved_documents or [])
            ],
            "relevance_scores": self.relevance_scores,
            "faithfulness_score": self.faithfulness_score,
            "answer_relevance_score": self.answer_relevance_score,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "latency_seconds": self.latency_seconds
        }

class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems,
    measuring performance across multiple dimensions.
    """
    
    def __init__(self, rag_pipeline, 
                eval_llm: Optional[Any] = None,
                metrics: List[str] = config.EVAL_METRICS):
        """
        Initialize the evaluator
        
        Args:
            rag_pipeline: The RAG pipeline to evaluate
            eval_llm: Optional LLM to use for evaluation, separate from the RAG pipeline
            metrics: List of metrics to evaluate
        """
        self.rag_pipeline = rag_pipeline
        
        # Initialize evaluation LLM if not provided
        if eval_llm is None:
            logger.info("Initializing evaluation LLM")
            self.eval_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0.0,  # Use deterministic outputs for evaluation
                api_key=config.OPENAI_API_KEY
            )
        else:
            self.eval_llm = eval_llm
        
        self.metrics = metrics
        self.evaluation_results = []
    
    def evaluate_retrieval(self, 
                           question: str, 
                           answer: str,
                           retrieved_documents: List[Document],
                           expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieved documents
        
        Args:
            question: The question being answered
            answer: The answer provided by the system
            retrieved_documents: The documents retrieved for the question
            expected_answer: Optional expected correct answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not retrieved_documents:
            logger.warning("No documents retrieved for evaluation")
            return {
                "relevance_scores": [],
                "context_precision": 0.0,
                "context_recall": 0.0
            }
        
        # Evaluate document relevance to the question
        relevance_scores = self._evaluate_document_relevance(
            question, retrieved_documents
        )
        
        # Calculate precision (% of retrieved docs that are relevant)
        # We consider a document relevant if its score is >= 0.5
        relevant_docs = sum(1 for score in relevance_scores if score >= 0.5)
        context_precision = relevant_docs / len(retrieved_documents) if retrieved_documents else 0
        
        # For recall, we would need to know all relevant documents in the corpus
        # This is usually approximated or requires ground truth data
        # Here we set it to None or can use a placeholder value
        context_recall = None
        
        return {
            "relevance_scores": relevance_scores,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
    
    def evaluate_answer(self, 
                        question: str, 
                        answer: str,
                        retrieved_documents: List[Document],
                        expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer
        
        Args:
            question: The question being answered
            answer: The answer provided by the system
            retrieved_documents: The documents retrieved for the question
            expected_answer: Optional expected correct answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate faithfulness (does the answer only use facts from the context?)
        faithfulness_score = self._evaluate_faithfulness(
            answer, retrieved_documents
        )
        
        # Evaluate answer relevance to the question
        answer_relevance_score = self._evaluate_answer_relevance(
            question, answer
        )
        
        return {
            "faithfulness_score": faithfulness_score,
            "answer_relevance_score": answer_relevance_score
        }
    
    def _evaluate_document_relevance(self, 
                                   question: str, 
                                   documents: List[Document]) -> List[float]:
        """
        Evaluate how relevant each document is to the question
        
        Args:
            question: The question
            documents: List of retrieved documents
            
        Returns:
            List of relevance scores (0-1) for each document
        """
        prompt_template = """
        You are an expert at evaluating whether documents are relevant to a question.
        
        ## Question:
        {question}
        
        ## Document:
        {document}
        
        ## Instructions:
        Evaluate how relevant this document is to answering the question.
        
        - Score 0: Not relevant at all, containing no useful information for the question.
        - Score 0.25: Slightly relevant, containing background information but not directly useful.
        - Score 0.5: Moderately relevant, containing some useful information but incomplete.
        - Score 0.75: Highly relevant, containing most information needed to answer the question.
        - Score 1.0: Perfectly relevant, containing all information needed to answer the question.
        
        ## Relevance Score (return only a number from 0, 0.25, 0.5, 0.75, or 1.0):
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "document"]
        )
        
        relevance_chain = LLMChain(llm=self.eval_llm, prompt=prompt)
        
        relevance_scores = []
        
        for doc in documents:
            try:
                score_str = relevance_chain.run(
                    question=question, 
                    document=doc.page_content
                ).strip()
                
                # Extract the numeric score
                try:
                    score = float(score_str)
                    # Ensure the score is in the valid range
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    logger.warning(f"Could not parse relevance score: {score_str}")
                    score = 0.0
                
                relevance_scores.append(score)
            except Exception as e:
                logger.error(f"Error evaluating document relevance: {str(e)}")
                relevance_scores.append(0.0)
        
        return relevance_scores
    
    def _evaluate_faithfulness(self, 
                             answer: str, 
                             documents: List[Document]) -> float:
        """
        Evaluate how faithful the answer is to the retrieved documents
        (i.e., does it only use facts from the context?)
        
        Args:
            answer: The answer to evaluate
            documents: The retrieved documents used for the answer
            
        Returns:
            Faithfulness score (0-1)
        """
        # Combine document content
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        prompt_template = """
        You are an expert at evaluating whether an answer is faithful to the provided context.
        
        ## Context from Retrieved Documents:
        {context}
        
        ## Answer:
        {answer}
        
        ## Instructions:
        Evaluate whether the answer is faithful to the context from the retrieved documents.
        The answer should only contain information that is explicitly stated in or can be directly
        inferred from the context. The answer should not contain any hallucinations or made-up facts.
        
        - Score 0: Completely unfaithful, containing major hallucinations or made-up facts.
        - Score 0.25: Mostly unfaithful, containing some hallucinations mixed with few correct facts.
        - Score 0.5: Partially faithful, containing some correct facts but also some hallucinations.
        - Score 0.75: Mostly faithful, containing mostly correct facts with minor inaccuracies.
        - Score 1.0: Completely faithful, containing only facts from the context with no hallucinations.
        
        ## Faithfulness Score (return only a number from 0, 0.25, 0.5, 0.75, or 1.0):
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "answer"]
        )
        
        faithfulness_chain = LLMChain(llm=self.eval_llm, prompt=prompt)
        
        try:
            score_str = faithfulness_chain.run(
                context=combined_content, 
                answer=answer
            ).strip()
            
            # Extract the numeric score
            try:
                score = float(score_str)
                # Ensure the score is in the valid range
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse faithfulness score: {score_str}")
                score = 0.0
            
            return score
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {str(e)}")
            return 0.0
    
    def _evaluate_answer_relevance(self, 
                                 question: str, 
                                 answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question
        
        Args:
            question: The question
            answer: The answer to evaluate
            
        Returns:
            Answer relevance score (0-1)
        """
        prompt_template = """
        You are an expert at evaluating whether an answer is relevant to a question.
        
        ## Question:
        {question}
        
        ## Answer:
        {answer}
        
        ## Instructions:
        Evaluate how relevant this answer is to the question, regardless of its factual accuracy.
        The answer should address what the question is asking about.
        
        - Score 0: Not relevant at all, completely fails to address the question.
        - Score 0.25: Slightly relevant, but mostly off-topic or missing the point of the question.
        - Score 0.5: Moderately relevant, addresses some aspects of the question but misses others.
        - Score 0.75: Highly relevant, addresses most aspects of the question directly.
        - Score 1.0: Perfectly relevant, directly and completely addresses all aspects of the question.
        
        ## Relevance Score (return only a number from 0, 0.25, 0.5, 0.75, or 1.0):
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "answer"]
        )
        
        relevance_chain = LLMChain(llm=self.eval_llm, prompt=prompt)
        
        try:
            score_str = relevance_chain.run(
                question=question, 
                answer=answer
            ).strip()
            
            # Extract the numeric score
            try:
                score = float(score_str)
                # Ensure the score is in the valid range
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse answer relevance score: {score_str}")
                score = 0.0
            
            return score
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return 0.0
    
    def evaluate_question(self, 
                         question: str, 
                         expected_answer: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate the RAG pipeline on a single question
        
        Args:
            question: The question to evaluate
            expected_answer: Optional expected correct answer
            
        Returns:
            EvaluationResult with metrics
        """
        logger.info(f"Evaluating question: {question}")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        # Get the answer from the RAG pipeline
        try:
            result = self.rag_pipeline.query(question)
            answer = result["answer"]
            retrieved_documents = result.get("source_documents", [])
            
            # Record end time and calculate latency
            latency = time.time() - start_time
            
            # Evaluate retrieval quality
            retrieval_metrics = self.evaluate_retrieval(
                question, answer, retrieved_documents, expected_answer
            )
            
            # Evaluate answer quality
            answer_metrics = self.evaluate_answer(
                question, answer, retrieved_documents, expected_answer
            )
            
            # Create evaluation result
            eval_result = EvaluationResult(
                question=question,
                answer=answer,
                expected_answer=expected_answer,
                retrieved_documents=retrieved_documents,
                relevance_scores=retrieval_metrics["relevance_scores"],
                faithfulness_score=answer_metrics["faithfulness_score"],
                answer_relevance_score=answer_metrics["answer_relevance_score"],
                context_precision=retrieval_metrics["context_precision"],
                context_recall=retrieval_metrics["context_recall"],
                latency_seconds=latency
            )
            
            # Save the evaluation result
            self.evaluation_results.append(eval_result)
            
            return eval_result
        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            
            # Create a failure result
            return EvaluationResult(
                question=question,
                answer="ERROR: " + str(e),
                expected_answer=expected_answer
            )
    
    def evaluate_dataset(self, 
                        questions: List[str], 
                        expected_answers: Optional[List[str]] = None) -> List[EvaluationResult]:
        """
        Evaluate the RAG pipeline on a dataset of questions
        
        Args:
            questions: List of questions to evaluate
            expected_answers: Optional list of expected answers
            
        Returns:
            List of EvaluationResult with metrics
        """
        logger.info(f"Evaluating dataset with {len(questions)} questions")
        
        results = []
        
        for i, question in enumerate(questions):
            expected_answer = expected_answers[i] if expected_answers and i < len(expected_answers) else None
            result = self.evaluate_question(question, expected_answer)
            results.append(result)
        
        return results
    
    def summarize_results(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics over all evaluation results
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        metrics = {}
        
        # Calculate average faithfulness
        faithfulness_scores = [
            r.faithfulness_score for r in self.evaluation_results 
            if r.faithfulness_score is not None
        ]
        if faithfulness_scores:
            metrics["avg_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
        
        # Calculate average answer relevance
        relevance_scores = [
            r.answer_relevance_score for r in self.evaluation_results 
            if r.answer_relevance_score is not None
        ]
        if relevance_scores:
            metrics["avg_answer_relevance"] = sum(relevance_scores) / len(relevance_scores)
        
        # Calculate average context precision
        precision_scores = [
            r.context_precision for r in self.evaluation_results 
            if r.context_precision is not None
        ]
        if precision_scores:
            metrics["avg_context_precision"] = sum(precision_scores) / len(precision_scores)
        
        # Calculate average document relevance
        doc_relevance = []
        for result in self.evaluation_results:
            if result.relevance_scores:
                doc_relevance.append(sum(result.relevance_scores) / len(result.relevance_scores))
        
        if doc_relevance:
            metrics["avg_document_relevance"] = sum(doc_relevance) / len(doc_relevance)
        
        # Calculate average latency
        latencies = [
            r.latency_seconds for r in self.evaluation_results 
            if r.latency_seconds is not None
        ]
        if latencies:
            metrics["avg_latency"] = sum(latencies) / len(latencies)
        
        # Calculate overall score (weighted average of other metrics)
        # This is just one possible way to combine metrics
        if all(key in metrics for key in ["avg_faithfulness", "avg_answer_relevance", "avg_context_precision"]):
            metrics["overall_score"] = (
                metrics["avg_faithfulness"] * 0.4 +
                metrics["avg_answer_relevance"] * 0.4 +
                metrics["avg_context_precision"] * 0.2
            )
        
        # Count total evaluations
        metrics["total_evaluations"] = len(self.evaluation_results)
        
        return metrics
    
    def save_results(self, output_file: str):
        """
        Save evaluation results to a JSON file
        
        Args:
            output_file: Path to the output file
        """
        results_dict = {
            "results": [r.to_dict() for r in self.evaluation_results],
            "summary": self.summarize_results()
        }
        
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2) 