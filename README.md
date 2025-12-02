# Accelerated-Large-Language-Model-Pre-Training

## Overview
This project introduces a novel approach to accelerate the pre-training process of Large Language Models (LLMs). Since modern LLMs contain billions of parameters, traditional (“vanilla”) pre-training is slow and requires significant computational resources.  
Our solution focuses on **dataset splitting**, **layer-based training**, and **parallel sub-model merging**, providing faster training with minimal performance loss.

## Project Goals
- Reduce LLM pre-training time.
- Train different model layers on different dataset chunks in parallel.
- Merge sub-models efficiently using **MergeKit**.
- Preserve performance across reasoning, commonsense, and math benchmarks.

## Methodology
1. **Base Model:** Llama-3.2-1B  
2. **Datasets:**  
   - Cosmopedia  
   - GSM8K Train  
   - MetaMathQA  
3. **Process:**  
   - Dataset split into four chunks  
   - Each chunk fine-tunes specific model layers  
   - Models trained in parallel using PyTorch  
   - Sub-models merged (SLERP, linear, TIES methods)  
4. **Evaluation:**  
   - ARC Challenge  
   - HellaSwag  
   - GSM8K (flexible & strict)  
   - Validation Loss (small & large)

## Key Results
- **SLERP-4** proved to be the best-performing strategy.
- Achieved **19–25% reduction** in training time on large datasets.
- Retained **50–90%** of vanilla fine-tuning gains depending on dataset.
- Successfully showed that parallel layer-based training is a viable alternative to standard pre-training.
