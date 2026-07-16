from transformers import AutoTokenizer

from config import MODEL_NAME


class Tokenizer:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Some models don't define a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str):

        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def decode(self, token_ids):

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

    def batch_encode(self, texts):

        return self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )

    def batch_decode(self, batch_ids):

        return self.tokenizer.batch_decode(
            batch_ids,
            skip_special_tokens=True
        )

    def count_tokens(self, text: str):

        return len(self.encode(text))

    def get_hf_tokenizer(self):
        """
        Expose the underlying HF tokenizer when needed
        (e.g. streamer, chat template).
        """
        return self.tokenizer