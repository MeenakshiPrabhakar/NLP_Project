## README

## Project Overview
This project trains a domain-specific BERT tokenizer using the Financial PhraseBank dataset. The goal is to improve tokenization for financial terminology by creating a vocabulary tailored to financial news and sentiment analysis tasks. 

Two tokenizers were created:
1. Financial-Specific Tokenizer
    - Re-trains the BERT tokenizer vocabulary using financial sentences only.
    - Output directory: finbert_financial_tokenizer
    - Source script: customtok_fin.py
3. Financial General Tokenizer
   - Keeps the original vocabulary but filters for financially relevant words
   - Ensures frequent domain words are included (greater than min frequency threshold)
   - Output directory: finbert_financial_general_tokenizer
   - Source script: customtok_fin_and_gen
  

## Dataset
We used the "sentences_50agree" split of the Financial PhraseBank:
- Source: takala/financial_phrasebank
- Only train + validation splits are used to construct vocabulary
- Short sentences (less than 20 characters) are filtered out for better training quality

## How to Run
To train the financial-specific tokenizer:
- run python customtok_fin.py

To train the general financial tokenizer:
- python customtok_fin_and_gen.py

Tokenizer files will be saved to their respective directories

## How it Works
1. Load Dataset: uses load_dataset from HuggingFace
2. Extract Corpus: iterates only through real financial text
3. Tokenizer training: adapts the existing BERT tokenizer
4. Save Vocabulary: outputs tokenizer files for later use



Notebook for custom tokenizer: https://colab.research.google.com/drive/1baEPxfzzj2QT2Sb32E_p6K8kGvwY17xI?usp=sharing
