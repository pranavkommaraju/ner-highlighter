import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vectors
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from model import NERModel

# Load Data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = load_dataset("conll2003", split="train")

tokens_list = train_dataset['tokens']
char_mapping = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
    't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28,
    '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36, '.': 37,
    ',': 38, "'": 39, '-': 40, ':': 41, ';': 42, '?': 43, '!': 44, '<pad>': 0, '<unk>': 45
}

max_char_length = 113

inputs = tokenizer(
    tokens_list,
    padding='max_length',
    truncation=True,
    max_length=113,
    return_tensors='pt',
    is_split_into_words=True
)
inputs = inputs['input_ids']

# Entity Labels
entity_labels = train_dataset['ner_tags']
max_label_entity = max(max(labels) for labels in entity_labels)
num_entity_labels = max_label_entity + 1
entity_labels = [torch.tensor(labels, dtype=torch.long) for labels in entity_labels]
entity_labels = pad_sequence(entity_labels, batch_first=True, padding_value=max_label_entity)

# Boundary Labels
boundary_labels = train_dataset['chunk_tags']
max_label_boundary = max(max(labels) for labels in boundary_labels)
num_boundary_labels = max_label_boundary + 1
boundary_labels = [torch.tensor(labels, dtype=torch.long) for labels in boundary_labels]
boundary_labels = pad_sequence(boundary_labels, batch_first=True, padding_value=max_label_boundary)

# Train Data
train_data = TensorDataset(inputs, entity_labels, boundary_labels)
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Hyperparameters + Model Initialization
word_vectors = Vectors("word_vecs.txt")
char_embedding_dim = 50
char_count = len(char_mapping)
hidden_dim = 200
dropout_rate = 0.5
alpha = 0.2
model = NERModel(word_vectors, char_embedding_dim, char_count, hidden_dim, num_entity_labels, num_boundary_labels, dropout_rate, alpha)

# Training Loop
num_epochs = 8
learning_rate = 0.005
max_grad_norm = 5
patience_for_stop = 3
best_loss = float('inf')
epochs_without_improvement = 0
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    model.train()
    total_loss = 0.0
    
    for batch_inputs, batch_entity_labels, batch_boundary_labels in train_data_loader:
        optimizer.zero_grad()
        
        batch_char_input_ids_list = []
        max_batch_char_length = max(len(token_chars) for tokens in batch_inputs for token_chars in tokenizer.convert_ids_to_tokens(tokens.tolist()))

        for token_ids in batch_inputs:
            tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
            token_char_sequences = [list(token) for token in tokens]
            batch_char_input_ids = []
            for token_chars in token_char_sequences:
                char_ids = [char_mapping.get(char, char_mapping['<unk>']) for char in token_chars]
                char_ids_padded = char_ids + [0] * (max_batch_char_length - len(char_ids))
                batch_char_input_ids.append(char_ids_padded)

            batch_char_input_ids_list.append(batch_char_input_ids)

        batch_char_input_ids = torch.tensor(batch_char_input_ids_list, dtype=torch.long)

        entity_probs, boundary_probs, multitask_loss = model(batch_inputs, batch_char_input_ids, batch_entity_labels, batch_boundary_labels, num_entity_labels, num_boundary_labels)
        multitask_loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += multitask_loss.item()
    
    avg_loss = total_loss / len(train_data_loader)
    print(f"Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience_for_stop:
        print("Early stopping! No improvement for", patience_for_stop, "epochs.")
        break

    scheduler.step(avg_loss)

torch.save(model.state_dict(), "trained_model.pth")