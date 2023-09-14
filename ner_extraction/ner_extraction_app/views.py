import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from transformers import BertTokenizer
from torchtext.vocab import Vectors
import time

import os
import sys

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(grandparent_dir)

from model import NERModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_mapping = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
    't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28,
    '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36, '.': 37,
    ',': 38, "'": 39, '-': 40, ':': 41, ';': 42, '?': 43, '!': 44, '<pad>': 0, '<unk>': 45
}
max_char_length = 113
word_vectors = Vectors("word_vecs.txt")
char_embedding_dim = 50
char_count = len(char_mapping)
hidden_dim = 200
num_entity_labels = 9
num_boundary_labels = 23
dropout_rate = 0.5
alpha = 0.2
model = NERModel(word_vectors, char_embedding_dim, char_count, hidden_dim, num_entity_labels, num_boundary_labels, dropout_rate, alpha)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@csrf_exempt
def extract_entities(request):
    if request.method == "POST":
        start_time = time.time()
        data = JSONParser().parse(request)
        text = data["text"]
        
        tokens_list = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_list)

        max_segment_length = 113
        segments = [input_ids[i:i+max_segment_length] for i in range(0, len(input_ids), max_segment_length)]

        entity_labels_list = []

        for segment in segments:
            char_input_ids_list = []
            for token_id in segment:
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                char_ids = [char_mapping.get(char, char_mapping['<unk>']) for char in token]
                char_ids_truncated = char_ids[:max_char_length]
                padding = [0] * (max_char_length - len(char_ids_truncated))
                char_input_ids_list.append(char_ids_truncated + padding)
            char_input_ids = torch.tensor(char_input_ids_list)

            input_ids = torch.tensor(segment).unsqueeze(0)
            char_input_ids = char_input_ids.unsqueeze(0)

            entity_labels = torch.zeros(input_ids.size(0), input_ids.size(1), num_entity_labels)
            boundary_labels = torch.zeros(input_ids.size(0), input_ids.size(1), num_boundary_labels)

            with torch.no_grad():
                entity_probs, _, _ = model(input_ids, char_input_ids, entity_labels, boundary_labels, num_entity_labels, num_boundary_labels)
            
            threshold = 0.9
            entity_labels_batch = process_predictions(entity_probs, threshold)
            entity_labels_list.extend(entity_labels_batch)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")
        return JsonResponse({"entities": entity_labels_list})

    return JsonResponse({"message": "Only POST requests are supported."})


entities = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

def process_predictions(entity_probs, threshold):
    print(entity_probs)
    entity_probs_binary = (entity_probs > threshold).float()
    print(entity_probs_binary)
    predicted_labels = []
    for batch_probs in entity_probs_binary:
        batch_labels = []
        for token_probs in batch_probs:
            entity_idx = torch.nonzero(token_probs).squeeze(dim=-1)
            if entity_idx.numel() == 0:
                batch_labels.append(0)
            else:
                entity_label_idx = entity_idx[0].item()
                batch_labels.append(entity_label_idx)
        
        predicted_labels.append(batch_labels)