# Parameter-Efficient Fine-Tuning and Vector Embeddings

This document provides comprehensive information about Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and QLoRA, LLaMA model specifications, and vector embedding considerations in our RAG system.

## LoRA (Low-Rank Adaptation)

### What is LoRA?

LoRA is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters by adding low-rank decomposition matrices to the model's weight matrices instead of fine-tuning all parameters.

### Key Concepts

- **Low-Rank Decomposition**: Instead of updating weight matrix W, LoRA decomposes updates into two smaller matrices: W + ΔW = W + BA, where B and A are low-rank matrices
- **Rank Dimension (r)**: Controls the expressivity vs. parameter efficiency tradeoff (typically r = 4, 8, 16, or 32)
- **Alpha Parameter**: Scaling factor that controls the magnitude of LoRA updates
- **Trainable Parameters Reduction**: LoRA can reduce trainable parameters by 10,000x while maintaining performance

### Example Implementation with PEFT Library

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,                       # Rank dimension
    lora_alpha=32,             # Alpha parameter scaling
    target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
    lora_dropout=0.05,         # Dropout probability for LoRA layers
    bias="none",               # Add bias to LoRA layers ("none", "all", or "lora_only")
    task_type="CAUSAL_LM"      # Task type for the model
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Apply LoRA to the model
lora_model = get_peft_model(base_model, lora_config)

# Print trainable parameters
print(f"Total parameters: {lora_model.num_parameters()}")
print(f"Trainable parameters: {lora_model.num_parameters(True)}")
```

### Benefits of LoRA

1. **Memory Efficiency**: Significantly reduces GPU memory requirements
2. **Training Speed**: Faster training due to fewer parameters to update
3. **Modularity**: Multiple LoRA adapters can be trained for different tasks on the same base model
4. **Model Merging**: Easier to merge multiple fine-tuned adaptations

## QLoRA (Quantized LoRA)

### What is QLoRA?

QLoRA combines quantization with LoRA to further reduce memory requirements, enabling fine-tuning of large language models on consumer hardware.

### Key Innovations

1. **4-bit Quantization**: Base model weights are quantized to 4-bit precision
2. **Double Quantization**: Further compressing memory footprint by quantizing the quantization constants
3. **NF4 Format**: "Normal Float 4-bit" data type optimized for weight distributions
4. **Paged Optimizers**: Memory management techniques to handle large models

### Implementation Example with bitsandbytes and PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Define 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model in 4-bit precision
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for kbit training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)
```

### QLoRA Performance Comparison

| Method | GPU Memory | Training Time | Performance |
|--------|------------|---------------|-------------|
| Full Fine-tuning (16-bit) | 120+ GB | 1x | Baseline |
| LoRA (16-bit) | 30-60 GB | 1.2x | Comparable |
| QLoRA (4-bit) | 10-12 GB | 1.6x | Near-SOTA |

### Memory Requirements for Fine-tuning LLaMA-2 with QLoRA

| Model Size | Memory with QLoRA | Consumer GPU |
|------------|-------------------|--------------|
| 7B | ~6-8 GB | RTX 3090, 4060+ |
| 13B | ~10-12 GB | RTX 3090, 4080+ |
| 70B | ~48 GB | A100, multiple GPUs |

## Fine-tuning LLaMA with LoRA/QLoRA

### Step-by-Step Process

1. **Data Preparation**:
   ```python
   from datasets import load_dataset
   
   # Load or prepare your dataset
   dataset = load_dataset("your_dataset")
   
   # Format data for instruction tuning
   def format_instruction(example):
       return {
           "text": f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
       }
   
   formatted_dataset = dataset.map(format_instruction)
   ```

2. **Training Configuration**:
   ```python
   from transformers import TrainingArguments
   
   training_args = TrainingArguments(
       output_dir="./llama-lora-output",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       gradient_checkpointing=True,
       optim="paged_adamw_8bit",
       learning_rate=2e-4,
       lr_scheduler_type="cosine",
       warmup_ratio=0.05,
       weight_decay=0.001,
       fp16=True,
       logging_steps=10,
       evaluation_strategy="steps",
       eval_steps=100,
       save_strategy="steps",
       save_steps=100,
       save_total_limit=3,
       load_best_model_at_end=True,
   )
   ```

3. **Training Loop**:
   ```python
   from transformers import Trainer, DataCollatorForLanguageModeling
   
   # Define data collator
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
   tokenizer.pad_token = tokenizer.eos_token
   data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
   
   # Initialize trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=formatted_dataset["train"],
       eval_dataset=formatted_dataset["validation"],
       data_collator=data_collator,
   )
   
   # Start training
   trainer.train()
   
   # Save LoRA adapter only (not the full model)
   model.save_pretrained("./llama-lora-adapter")
   ```

4. **Inference with Fine-tuned Model**:
   ```python
   from peft import PeftModel, PeftConfig
   
   # Load base model
   base_model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-7b-hf",
       torch_dtype=torch.float16,
       device_map="auto",
   )
   
   # Load LoRA adapter
   model = PeftModel.from_pretrained(base_model, "./llama-lora-adapter")
   
   # Generate text
   inputs = tokenizer("### Instruction: Explain quantum computing in simple terms\n### Input:\n### Response:", return_tensors="pt").to("cuda")
   outputs = model.generate(inputs["input_ids"], max_new_tokens=500)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

## LLaMA Model Specifications

### Model Sizes and Parameters

| LLaMA Version | Variants | Parameter Count | Model Size (FP16) | Training Tokens |
|---------------|----------|-----------------|-------------------|-----------------|
| LLaMA 1       | 7B, 13B, 33B, 65B | 7-65 billion | 13-120 GB | 1.4 trillion |
| LLaMA 2       | 7B, 13B, 70B | 7-70 billion | 13-140 GB | 2 trillion |
| LLaMA 3       | 8B, 70B | 8-70 billion | 16-140 GB | 15 trillion |

### Context Window Lengths

| LLaMA Version | Base Context Length | Extended Context |
|---------------|---------------------|------------------|
| LLaMA 1       | 2,048 tokens        | 4,096+ with techniques |
| LLaMA 2       | 4,096 tokens        | Up to 8,192 with fine-tuning |
| LLaMA 3       | 8,192 tokens        | Up to 128,000 tokens |

In our RAG system, we use LLaMA 2 13B with a context window of 4,096 tokens that we've extended to 8,192 tokens through position interpolation techniques.

## Vector Embeddings in Our RAG System

### Embedding Model and Dimensions

Our RAG system uses the following embedding specifications:

- **Embedding Model**: `intfloat/e5-large-v2`
- **Embedding Dimension**: 1,024 dimensions per embedding
- **Context Window**: 512 tokens per embedding input
- **Normalization**: L2 normalization is applied to all embeddings

### Vector Store Volume

Our production RAG system manages:

- **Total Documents**: ~1.2 million documents
- **Average Chunks per Document**: 8.5 chunks
- **Total Vector Embeddings**: ~10 million vectors
- **Storage Size**: 42 GB (embeddings only), 65 GB (with metadata)
- **Embedding Update Frequency**: Daily incremental updates, weekly full refresh

### Chunking Strategy

We employ a hybrid chunking strategy:

```python
def create_semantic_chunks(document, chunk_size=512, chunk_overlap=50):
    # Semantic chunking implementation
    semantic_chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = semantic_chunker.chunk_document(document)
    
    # Further processing for improved retrieval
    processed_chunks = []
    for chunk in chunks:
        # Add metadata for improved retrieval
        chunk.metadata.update({
            "document_id": document.id,
            "section": extract_section(chunk.text),
            "chunk_size_tokens": count_tokens(chunk.text),
            "created_at": datetime.now().isoformat(),
        })
        processed_chunks.append(chunk)
    
    return processed_chunks
```

## Is a Vector Store Mandatory for RAG?

**No, a vector store is not mandatory for RAG systems**, although it is the most common and efficient implementation.

### Alternative Approaches to Vector Stores

1. **In-Memory Embeddings**: For small document collections, embeddings can be stored in memory and similarity search performed on-the-fly.

    ```python
    class InMemoryRetriever:
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.documents = []
            self.embeddings = []
        
        def add_documents(self, documents):
            self.documents.extend(documents)
            embeddings = self.embedding_model.encode([doc.text for doc in documents])
            self.embeddings.extend(embeddings)
        
        def search(self, query, k=5):
            query_embedding = self.embedding_model.encode(query)
            
            # Compute similarities
            similarities = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
            
            # Sort and return top k
            sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
            return [self.documents[i] for i in sorted_indices]
    ```

2. **Sparse Retrieval Methods**: Traditional IR techniques like BM25 can be used in RAG:

    ```python
    from rank_bm25 import BM25Okapi
    
    class BM25Retriever:
        def __init__(self):
            self.documents = []
            self.bm25 = None
            self.tokenized_documents = []
        
        def add_documents(self, documents):
            self.documents.extend(documents)
            self.tokenized_documents = [doc.text.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_documents)
        
        def search(self, query, k=5):
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self.documents[i] for i in sorted_indices]
    ```

3. **Hybrid Retrievers**: Combining multiple retrieval methods:

    ```python
    class HybridRetriever:
        def __init__(self, vector_retriever, sparse_retriever, alpha=0.5):
            self.vector_retriever = vector_retriever
            self.sparse_retriever = sparse_retriever
            self.alpha = alpha
        
        def search(self, query, k=5):
            # Get results from both retrievers
            vector_results = self.vector_retriever.search(query, k=k*2)
            sparse_results = self.sparse_retriever.search(query, k=k*2)
            
            # Merge and deduplicate results
            results_dict = {}
            
            for doc in vector_results:
                results_dict[doc.id] = {"doc": doc, "vector_score": doc.score, "sparse_score": 0}
            
            for doc in sparse_results:
                if doc.id in results_dict:
                    results_dict[doc.id]["sparse_score"] = doc.score
                else:
                    results_dict[doc.id] = {"doc": doc, "vector_score": 0, "sparse_score": doc.score}
            
            # Compute hybrid scores
            for doc_id, data in results_dict.items():
                data["hybrid_score"] = (self.alpha * data["vector_score"] + 
                                        (1 - self.alpha) * data["sparse_score"])
            
            # Sort by hybrid score and return top k
            sorted_results = sorted(results_dict.values(), 
                                    key=lambda x: x["hybrid_score"], reverse=True)[:k]
            return [item["doc"] for item in sorted_results]
    ```

### When Vector Stores Are Essential

Vector stores become essential when:

1. **Scale**: Managing millions of vectors efficiently
2. **Performance**: Requiring sub-second query latency
3. **Filtering**: Needing complex metadata filtering capabilities
4. **Updates**: Requiring frequent index updates

In our RAG system, we use Chroma as our vector database because of its superior performance with filtering operations and its integration with our existing stack. For specific applications where volume is lower, we have successfully implemented RAG with simpler retrieval methods. 