import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Hyperparameters
EMBED_SIZE = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.bn(self.fc(features)))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, hidden, cell):
        embeddings = self.dropout(self.embed(captions))
        lstm_out, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        outputs = self.fc(lstm_out)
        return outputs, hidden, cell

class Seq2SeqCaptioner(nn.Module):
    def __init__(self, encoder, decoder, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        enc_out = self.encoder(features)
        hidden = enc_out.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        outputs, _, _ = self.decoder(captions, hidden, cell)
        return outputs

# Standard ResNet image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_resnet_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.to(DEVICE)
    extractor.eval()
    return extractor

@torch.no_grad()
def extract_feature(image: Image.Image, extractor):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    feature = extractor(tensor).view(1, -1)
    return feature

@torch.no_grad()
def generate_caption(model, feature, idx2word, beam_width=5, max_len=30):
    model.eval()
    enc_out = model.encoder(feature)
    hidden = enc_out.unsqueeze(0).repeat(model.num_layers, 1, 1)
    cell = torch.zeros_like(hidden)

    # Beam search: (score, token_list, hidden, cell)
    beams = [(0.0, [1], hidden, cell)]  # 1 is <start>
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, seq, h, c in beams:
            last_tok = torch.tensor([[seq[-1]]], device=DEVICE)
            output, h_new, c_new = model.decoder(last_tok, h, c)
            log_probs = torch.log_softmax(output.squeeze(0).squeeze(0), dim=0)
            topk_probs, topk_idx = log_probs.topk(beam_width)

            for k in range(beam_width):
                tok = topk_idx[k].item()
                new_seq = seq + [tok]
                new_score = score + topk_probs[k].item()
                if tok == 2:  # 2 is <end>
                    completed.append((new_score, new_seq))
                else:
                    candidates.append((new_score, new_seq, h_new, c_new))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(s, seq, h, c) for s, seq, h, c in candidates[:beam_width]]

    if not completed:
        completed = [(s, seq) for s, seq, _, _ in beams]

    completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
    best_tokens = completed[0][1]

    # Convert to words
    words = [idx2word.get(i, '<unk>') for i in best_tokens if i not in (0, 1, 2)]
    return ' '.join(words)
