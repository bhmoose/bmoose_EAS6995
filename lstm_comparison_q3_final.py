
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and process data
with open('input.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert text to numerical indices
def encode_text(text):
    return [char_to_ix[ch] for ch in text]

def decode_text(indices):
    return ''.join(ix_to_char[i] for i in indices)

# Define PyTorch LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout_rate=0.3):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state (h, c) for LSTM"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))
    
# Function to generate text
def sample(model, start_char, length=200):
    model.eval()
    input_char = torch.zeros(1, 1, vocab_size, device=device)
    input_char[0, 0, char_to_ix[start_char]] = 1

    hidden = model.init_hidden(batch_size=1)
    output_text = start_char

    for _ in range(length):
        output, hidden = model(input_char, hidden)
        probs = torch.softmax(output[0, 0], dim=0).detach().cpu().numpy()
        next_char = np.random.choice(range(vocab_size), p=probs)
        output_text += ix_to_char[next_char]
        input_char = torch.zeros(1, 1, vocab_size, device=device)
        input_char[0, 0, next_char] = 1

    return output_text


#Set model parameters
vocab_size = len(chars)
hidden_size = 100
seq_length = 25
num_layers = 2  # Number of LSTM layers
dropout_rate = 0
learning_rate = 0.01

num_epochs = 100
batch_size = 64

#Create files to store output
textfile = open('lstm_comparison_text.txt', 'w+')
lossfile = open('lstm_comparison_loss.txt', 'w+')

# Initialize model, loss function, optimizer
model = CharLSTM(vocab_size, hidden_size, num_layers, dropout_rate).to(device)  #Initialize model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Iterate over epochs
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(batch_size=batch_size)  #Initialize zero hidden state
    epoch_loss_total = 0 #Track loss for batch

    #Iterate over batches
    for i in range(len(data) // (batch_size*seq_length)): #i represents batch number
        inputs = np.zeros((batch_size, seq_length)) #Set up input array
        targets = np.zeros((batch_size, seq_length)) #Set up target array
        row_idx = 0
        #Iterate over characters in batch
        for j in range(i*batch_size*seq_length, i*batch_size*seq_length + (batch_size*seq_length), seq_length):
            #Encode all sequences in batch
            inputs[row_idx, :] = encode_text(data[j:j+seq_length])
            targets[row_idx, :] = encode_text(data[j+1:j+seq_length+1])
            row_idx = row_idx + 1


        #Create input and target tensors
        input_tensor = torch.zeros(batch_size, seq_length, vocab_size, device=device)
        target_tensor = torch.zeros(batch_size, seq_length, device=device)

        #Iterate through sequences and characters, fill in input and target tensors
        for seqidx in range(inputs.shape[0]):
            for charidx in range(inputs.shape[1]):
                charval = int(inputs[seqidx, charidx])
                targetval = int(targets[seqidx, charidx])
                input_tensor[seqidx, charidx, charval] = 1
                target_tensor[seqidx, charidx] = targetval
        
        optimizer.zero_grad()
        output, __ = model(input_tensor, hidden) #Run model with zero initial hidden state
        loss = criterion(output.view(-1, vocab_size), target_tensor.long().view(-1)) #Get loss for batch
        epoch_loss_total = epoch_loss_total + loss #Track epoch loss
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {epoch_loss_total / (len(data) // (batch_size*seq_length))}')

    #Write loss to lossfile
    lossfile.write(str(epoch_loss_total.item() / (len(data) // (batch_size*seq_length))))
    lossfile.write('\n')

    #Write sampled text to textfile
    if epoch % 10 == 0:
        sampletext = sample(model, start_char='H')
        print(sampletext)
        textfile.write(sampletext)
        textfile.write('\n')
        textfile.write('-------------------------')
        textfile.write('\n')

lossfile.close()
textfile.close()
