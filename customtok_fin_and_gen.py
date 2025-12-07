from collections import Counter
from datasets import load_dataset
from transformers import BertTokenizerFast

def load_financial_phrasebank(seed:int=42):
    raw = load_dataset("takala/financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    split = raw["train"].train_test_split(test_size=0.2, seed=seed)
    train_val = split["train"].train_test_split(test_size=0.125, seed=seed)
    split["train"] = train_val["train"]
    split["validation"] = train_val["test"]
    return split

def iter_financial_text(dataset):
    for split_name in ["train", "validation"]:
        for row in dataset[split_name]:
            sentence = row["sentence"].strip()
            if len(sentence) > 20:
                yield sentence

def train_financial_tokenizer(base_model_name, text_iterator, save_directory, min_freq: int = 5):
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
    word_counts = Counter()
    for text in text_iterator:
        for word in text.split():
            w = word.strip()
            if not w:
                continue
            word_counts[w] += 1
    print(f"unique raw tokens in corpus: {len(word_counts)}")
    candidate_tokens = []
    for word, freq in word_counts.items():
        if freq < min_freq:
            continue
        pieces = tokenizer.tokenize(word)
        if len(pieces) > 1 or (len(pieces) == 1 and pieces[0] == tokenizer.unk_token):
            candidate_tokens.append(word)
    tokenizer.save_pretrained(save_directory)
    return tokenizer

def main():
    dataset = load_financial_phrasebank(seed=42)
    financial_text_iterator = iter_financial_text(dataset)
    train_financial_tokenizer(
        base_model_name="bert-base-uncased",
        text_iterator=financial_text_iterator,
        save_directory="finbert_financial_general_tokenizer",
        min_freq=5,
    )

if __name__ == "__main__":
    main()
