# ner-highlighter
Boundary-Aware Named Entity Recognition Model  
The model uses the bert-base-uncased tokenizer and was trained on the conll2003 dataset.
It uses Word2vec word embeddings and a LSTM for char embeddings, then combines the two into a shared LSTM.
It contains an entity and boundary classifier for better recognition. 

The named entities it can capture are: "Person", "Location", "Organization", "Miscellaneous".
The model is hosted on a Django server so it can be called while searching the web, therefore highlighting named entities on Chrome webpages.
