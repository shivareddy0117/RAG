import logging
from typing import List, Dict, Any, Optional, Union
from langchain.prompts import PromptTemplate

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class PromptTemplates:
    """
    Advanced prompt templates for different use cases,
    optimized for RAG applications.
    """
    
    @staticmethod
    def qa_prompt() -> PromptTemplate:
        """
        Create a prompt template for question answering
        
        Returns:
            PromptTemplate for QA
        """
        template = """
        You are an AI assistant providing helpful, accurate, and concise information.
        
        ## Context:
        {context}
        
        ## Question:
        {question}
        
        ## Instructions:
        - Answer the question based only on the provided context.
        - If the context doesn't contain enough information to answer the question, admit that you don't know rather than making up information.
        - Provide a direct and concise answer.
        - Always cite the source documents when possible.
        - Use bullet points for lists or multiple points.
        
        ## Answer:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def summarization_prompt() -> PromptTemplate:
        """
        Create a prompt template for summarization
        
        Returns:
            PromptTemplate for summarization
        """
        template = """
        You are an AI assistant that creates concise and accurate summaries of documents.
        
        ## Document:
        {context}
        
        ## Instructions:
        - Provide a comprehensive summary of the above document.
        - Maintain the key facts, arguments, and important details.
        - Organize the summary in a coherent structure.
        - Focus on the most important information.
        - Keep the summary concise but thorough.
        - Use clear, direct language.
        
        ## Summary:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    @staticmethod
    def extraction_prompt(schema: Dict[str, Any]) -> PromptTemplate:
        """
        Create a prompt template for structured information extraction
        
        Args:
            schema: Schema of the information to extract
            
        Returns:
            PromptTemplate for extraction
        """
        # Convert schema to a readable format
        schema_str = "\n".join([
            f"- {key}: {value.get('description', '')}" 
            for key, value in schema.items()
        ])
        
        template = f"""
        You are an AI assistant that extracts specific information from documents.
        
        ## Document:
        {{context}}
        
        ## Information to Extract:
        {schema_str}
        
        ## Instructions:
        - Extract the requested information from the document.
        - Format the output as a JSON object matching the specified schema.
        - If information isn't present in the document, return null for that field.
        - Be precise and only extract information that is explicitly stated.
        
        ## Extracted Information (JSON):
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    @staticmethod
    def classification_prompt(categories: List[str]) -> PromptTemplate:
        """
        Create a prompt template for document classification
        
        Args:
            categories: List of categories for classification
            
        Returns:
            PromptTemplate for classification
        """
        categories_str = "\n".join([f"- {category}" for category in categories])
        
        template = f"""
        You are an AI assistant that classifies documents into predefined categories.
        
        ## Document:
        {{context}}
        
        ## Available Categories:
        {categories_str}
        
        ## Instructions:
        - Classify the document into one of the provided categories.
        - Provide a brief explanation for your classification.
        - If the document fits multiple categories, choose the most appropriate one.
        
        ## Classification:
        Category: 
        
        ## Explanation:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    @staticmethod
    def comparison_prompt() -> PromptTemplate:
        """
        Create a prompt template for comparing multiple documents
        
        Returns:
            PromptTemplate for document comparison
        """
        template = """
        You are an AI assistant that compares and analyzes multiple documents.
        
        ## Document 1:
        {document_1}
        
        ## Document 2:
        {document_2}
        
        ## Instructions:
        - Compare the two documents, identifying similarities and differences.
        - Analyze the key points, arguments, and facts presented in each.
        - Highlight any contradictions or agreement between the documents.
        - Provide an objective assessment without bias.
        
        ## Comparison:
        
        ### Similarities:
        
        ### Differences:
        
        ### Key Insights:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["document_1", "document_2"]
        )
    
    @staticmethod
    def fact_checking_prompt() -> PromptTemplate:
        """
        Create a prompt template for fact checking
        
        Returns:
            PromptTemplate for fact checking
        """
        template = """
        You are an AI assistant that carefully verifies factual claims against provided reference materials.
        
        ## Claim to Verify:
        {claim}
        
        ## Reference Material:
        {context}
        
        ## Instructions:
        - Assess whether the claim is supported, contradicted, or not addressed by the reference material.
        - Provide direct quotes from the reference material that support your assessment.
        - Make a clear judgment: SUPPORTED, CONTRADICTED, NOT ENOUGH INFORMATION, or PARTIALLY SUPPORTED.
        - Explain your reasoning in detail.
        - Do not use outside knowledge beyond the provided reference material.
        
        ## Assessment:
        
        Judgment: 
        
        Evidence:
        
        Explanation:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["claim", "context"]
        )
    
    @staticmethod
    def research_prompt() -> PromptTemplate:
        """
        Create a prompt template for research assistance
        
        Returns:
            PromptTemplate for research
        """
        template = """
        You are an AI research assistant helping to analyze academic materials and formulate insights.
        
        ## Research Question:
        {question}
        
        ## Relevant Materials:
        {context}
        
        ## Instructions:
        - Analyze the materials thoroughly in relation to the research question.
        - Synthesize key findings and insights from the materials.
        - Identify any gaps, contradictions, or limitations in the available information.
        - Suggest potential directions for further inquiry.
        - Maintain academic rigor and precision in your analysis.
        - Cite specific sources from the provided materials to support your analysis.
        
        ## Analysis:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )
    
    @staticmethod
    def multi_hop_reasoning_prompt() -> PromptTemplate:
        """
        Create a prompt template for multi-hop reasoning
        
        Returns:
            PromptTemplate for multi-hop reasoning
        """
        template = """
        You are an AI assistant skilled at complex reasoning across multiple pieces of information.
        
        ## Question Requiring Multi-Step Reasoning:
        {question}
        
        ## Available Information:
        {context}
        
        ## Instructions:
        - Break down the reasoning process into clear, logical steps.
        - Identify connections between different pieces of information.
        - Show your work by explaining each step of your reasoning process.
        - Draw only conclusions that are supported by the available information.
        - Be explicit about any assumptions you make.
        - If multiple reasoning paths are possible, explore the most promising one.
        
        ## Step-by-Step Reasoning:
        
        ## Final Answer:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )
    
    @staticmethod
    def retrieval_enhancement_prompt() -> PromptTemplate:
        """
        Create a prompt template for generating better search queries
        to improve retrieval performance.
        
        Returns:
            PromptTemplate for query enhancement
        """
        template = """
        You are an AI assistant that helps improve search queries to retrieve better information.
        
        ## Original Query:
        {query}
        
        ## Instructions:
        - Analyze the original query and identify its core information need.
        - Expand the query to include related terms, synonyms, and contextual information.
        - Break down complex queries into simpler, more focused sub-queries.
        - Rephrase the query to be more specific and targeted.
        - Add any missing context that would help retrieve more relevant information.
        - Format the output as a list of improved search queries.
        
        ## Enhanced Queries:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["query"]
        ) 