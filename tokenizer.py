import tiktoken


class Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self._vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
