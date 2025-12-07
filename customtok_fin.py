from datasets import load_dataset
from transformers import BertTokenizerFast

def load_financial_phrasebank(seed:int=42):
    financial_phrasebank_raw = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        trust_remote_code=True,
    )
    dataset = financial_phrasebank_raw["train"].train_test_split(
        test_size=0.2,
        seed=seed,
    )
    train_validation_split = dataset["train"].train_test_split(
        test_size=0.125,
        seed=seed,
    )
    dataset["train"] = train_validation_split["train"]
    dataset["validation"] = train_validation_split["test"]
    return dataset


def iter_financial_text(dataset):
    # only use train + validation to build this tokenizer
    for split_name in ["train", "validation"]:
        for row in dataset[split_name]:
            sentence = row["sentence"].strip()
            if len(sentence) > 20:
                yield sentence


def train_financial_tokenizer(base_model_name, text_iterator, save_directory):
    base_tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
    new_vocab_size = len(base_tokenizer)
    financial_tokenizer = base_tokenizer.train_new_from_iterator(text_iterator, vocab_size=new_vocab_size)
    financial_tokenizer.save_pretrained(save_directory)
    return financial_tokenizer


def main():
    dataset = load_financial_phrasebank(seed=42)
    financial_text_iterator = iter_financial_text(dataset)
    train_financial_tokenizer(
        base_model_name="bert-base-uncased",
        text_iterator=financial_text_iterator,
        save_directory="finbert_financial_tokenizer",
    )


if __name__ == "__main__":
    main()
