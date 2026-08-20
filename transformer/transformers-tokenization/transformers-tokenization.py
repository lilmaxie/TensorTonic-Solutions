from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        self.word_to_id = {}
        self.id_to_word = {}

        for idx, token in enumerate(self.special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        sorted_unique_words = sorted(list(unique_words))

        current_id = len(self.special_tokens)
        for word in sorted_unique_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(w, unk_id) for w in words]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        unk_token_str = self.unk_token
        words = [self.id_to_word.get(token_id, unk_token_str) for token_id in ids]
        return " ".join(words)