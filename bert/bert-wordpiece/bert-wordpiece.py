from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # check if the word length exceeds the allowed limit
        if len(word) > self.max_word_len:
            return [self.unk_token]

        is_bad = False
        start = 0
        sub_tokens = []

        # scan from start to end of a word
        while start < len(word):
            end = len(word)
            cur_substr = None

            # gradually move the `end` pointer towards the `start` pointer to find the longest substring present in the vocab
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            # if no matching part is found in the dictionary
            if cur_substr is None:
                is_bad = True
                break

            # save the found part and move the start pointer
            sub_tokens.append(cur_substr)
            start = end

        # if there is a mismatch in the character string, return the token [UNK] for the entire word
        if is_bad:
            return [self.unk_token]

        return sub_tokens