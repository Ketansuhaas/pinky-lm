import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        return self.sp.DecodeIds(tokens)

    def __len__(self):
        return self.sp.GetPieceSize()
