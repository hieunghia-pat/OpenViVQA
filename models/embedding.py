from torch import nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim, d_model, dropout=0.5):
        super(TextEmbedding, self).__init__()

        self.embedding = nn.Embedding(len(vocab.stoi), embedding_dim, padding_idx=vocab.stoi["<pad>"])
        if vocab.vectors is not None:
            self.embedding.from_pretrained(vocab.vectors)
        self.proj = nn.Linear(embedding_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

    def forward(self, x):
        x = self.proj(self.embedding(x))
        x = self.dropout(x)

        x, _ = self.lstm(x)

        return x

class VisualEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v):
        n, c, h, w = v.size()
        v = v.view(n, c, w*h).permute(0, 2, 1)

        v = self.proj(v)

        return v