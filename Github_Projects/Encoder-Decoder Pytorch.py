import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input = torch.randint(low=1, high=50, size=(50, 6),
                      dtype=torch.long).to(device)
target = input[:, :3]
target = torch.flip(target, (1, ))

target = torch.cat(
    (target,
     torch.zeros(
         target.size(0), target.size(1), device=device, dtype=torch.long)),
    dim=1)

train_dataset = torch.utils.data.TensorDataset(input, target)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)


class EncoderRNN(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size)

    def forward(self, x):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell


class DecoderRNN(nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        #shape of outputs (1, N , hidden_size)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


def train_model(source, target, encoder, decoder):
    hidden, cell = encoder(source)

    x = target[0]

    output, hidden, cell = decoder(x, hidden, cell)

    best_guess = output.argmax(1)
    #print("Best guess", best_guess[:3])

    return output


encoder = EncoderRNN(input_size=50, embedding_size=300,
                     hidden_size=1024).to(device)
decoder = DecoderRNN(input_size=50,
                     embedding_size=300,
                     output_size=50,
                     hidden_size=1024).to(device)
criterion = torch.nn.CrossEntropyLoss()

encoder_optmizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optmizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

for epoch in range(20):
    print("Epoch: ", epoch)
    total_loss = 0
    for data in train_dataloader:
        input_tensor, target_tensor = data
        output = train_model(input_tensor, target, encoder, decoder)
        #shape (target_len, batch_size, output_dim)

        target_tensor = target_tensor.squeeze(0)
        loss = criterion(output, target_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
        encoder_optmizer.step()
        decoder_optmizer.step()
        total_loss += loss.item()

    print("Loss: ", total_loss / len(train_dataloader))
