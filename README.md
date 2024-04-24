# Project Overview
This study investigates enhancing question-answering tasks by integrating Retrieval-Augmented Generation (RAG) models and Large Language Models (LLMs). It addresses challenges in LLMs including handling unseen data, lack of generalizability, and high fine-tuning costs. By leveraging a dataset from 270+ news websites, employing preprocessing techniques like NER and POS tagging, and using embeddings from HuggingFace and OpenAI, the study optimizes retrieval processes. It involves comparisons across RAG pipelines, evaluated with statistical metrics like ROUGE score, F1 score, and BLEU score, as well as non-statistical metrics and human evaluations. The approach promises robust handling of unseen data and domain-specific challenges, enhancing LLM adaptability and efficiency in diverse applications.

This repository contains code, dataset and response generation files for RAG based LLM implementation for Llama and GPT using Hugging Face and OpenAI Embedding models

# File Structure and Explaination
- scrape.py: contains code for scraping data from 270+ news websites 
- sitemap.xml: consists of URL of all the news websites. This was passed to the scraping script to extract data from
- scraped_data.txt: contains the scraped data in raw format consisting of the title of the webpage and the page content
- NLP_Project_Dataset.csv: contains all the questions made from the news websites along with the ground truth for them
requirements.txt: consists of all the python dependencies
- extracted_answers.csv: a demo module of how the generated responses from the RAG piepline along with the source documents would look like
- NLP_GPT_RAG_Pipeline.ipynb: jupyter notebook consisting of the entire pipeline starting with data collection, preprocessing by appling tagging techniques like NER, POS, NER+POS, applying vector embeddings (Hugging Face, OpenAI) and storing them in ChromaDB, using GPT LLM model to generate responses and store along with the ground truth
- NLP_Llama_RAG_Pipeline.ipynb: jupyter notebook consisting of the entire pipeline starting with data collection, preprocessing by appling tagging techniques like NER, POS, NER+POS, applying vector embeddings (Hugging Face, OpenAI) and storing them in ChromaDB, using Llama LLM model to generate responses and store along with the ground truth
- NLP_News_EvalProcessing.ipynb: jupyter notebook consisting of various statistical (ROUGE-L,METEOR,BLEU) and non-statistical (BERT,SBERT) methods used to determine the performance of the RAG pipeline
- RAGAS_NewsEval.ipynb: jupyter notebook to determine correctness, similarity, wrong answer count using non-statistical metric - RAGAS

# Steps to replicate
- Use the requirements.txt file to install all python dependencies
- Modify the sitemap.xml to add or remove website links
- Run the scraping script to begin scraping data from these websites
- Apply appropriate tagging techniques (NER, POS) using specific cells from NLP_GPT_RAG_Pipeline.ipynb
- To generate responses using Llama, use the NLP_Llama_Pipeline.ipynb and add the paths for the scraped data and preprocessed data. Model path for Llama was given directly, however, one could also use the API key for Llama
- To generate responses using GPT, use the NLP_Llama_Pipeline.ipynb and add the paths for the scraped data and preprocessed data. Make sure to give the OpenAPI key
- Use the NLP_News_EvalProcessing.ipynb and RAGAS_NewsEval.ipynb to evaluate your findings and responses
