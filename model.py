import torch
import torch.nn as nn
import torch.nn.functional as F

class NERModel(nn.Module):
    def __init__(self, word_vectors, char_embedding_dim, char_count, hidden_dim, num_entity_labels, num_boundary_labels, dropout_rate, alpha):
        super().__init__()

        # Word Embeddings
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors.vectors)
        self.word_embedding_dim = self.word_embedding.embedding_dim

        # Character Embeddings + LSTM
        self.char_embedding_dim = char_embedding_dim
        self.char_embedding = nn.Embedding(char_count, self.char_embedding_dim)
        self.char_lstm = nn.LSTM(self.char_embedding_dim, hidden_dim, bidirectional=True)
        # Concatenated LSTM
        self.shared_dim = self.word_embedding_dim + hidden_dim * 2
        self.shared_lstm = nn.LSTM(self.shared_dim, hidden_dim, bidirectional=True)

        # Entity Classifier
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_entity_labels),
            nn.ReLU(),
            nn.Linear(num_entity_labels, num_entity_labels)
        )

        # Boundary Classifier
        self.boundary_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_boundary_labels),
            nn.ReLU(),
            nn.Linear(num_boundary_labels, num_boundary_labels)
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.alpha = alpha

    def forward(self, input_ids, char_input_ids, entity_labels, boundary_labels, num_entity_labels, num_boundary_labels):
        # Feature Extraction
        word_embedded_input = self.word_embedding(input_ids)
        char_embedded_input = self.char_embedding(char_input_ids)
        batch_size, sequence_length, char_sequence_length, char_embedding_dim = char_embedded_input.size()
        char_embedded_input = char_embedded_input.view(batch_size * sequence_length, char_sequence_length, char_embedding_dim)
        char_lstm_output, _ = self.char_lstm(char_embedded_input)
        char_lstm_output = char_lstm_output.view(batch_size, sequence_length, char_sequence_length, self.char_lstm.hidden_size * 2)
        char_lstm_output_pooled, _ = char_lstm_output.max(dim=2)
        combined_embedded_input = torch.cat((word_embedded_input, char_lstm_output_pooled), dim=-1)

        lstm_output, _ = self.shared_lstm(combined_embedded_input)
        lstm_output = self.dropout(lstm_output)

        # Entity Categorical Label Prediction
        entity_logits = self.entity_classifier(lstm_output)
        entity_probs = torch.softmax(entity_logits, dim=-1)

        # Entity Boundary Detection
        boundary_logits = self.boundary_classifier(lstm_output)
        boundary_probs = torch.softmax(boundary_logits, dim=-1)

        # Multitask Training
        multitask_loss = self.multitask_loss(entity_probs, boundary_probs, entity_labels, boundary_labels, num_entity_labels, num_boundary_labels)
        return entity_probs, boundary_probs, multitask_loss
    
    def multitask_loss(self, entity_probs, boundary_probs, entity_labels, boundary_labels, num_entity_labels, num_boundary_labels):
        # Entity Loss
        entity_probs = entity_probs.view(-1, num_entity_labels)
        batch_size = entity_labels.size(0)
        seq_length = entity_labels.size(1)
        entity_labels = entity_labels.view(batch_size * seq_length, -1)
        entity_labels = entity_labels.squeeze()
        lecls_loss = F.cross_entropy(entity_probs, entity_labels)

        # Boundary Loss
        boundary_probs = boundary_probs.view(-1, num_boundary_labels)
        batch_size = boundary_labels.size(0)
        seq_length = boundary_labels.size(1)
        boundary_labels = boundary_labels.view(batch_size * seq_length, -1)
        boundary_labels = boundary_labels.squeeze()
        lbcls_loss = F.cross_entropy(boundary_probs, boundary_labels)


        # Multitask Loss
        multitask_loss = self.alpha * lbcls_loss + (1 - self.alpha) * lecls_loss
        return multitask_loss