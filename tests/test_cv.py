from transformers import DistilBertModel, DistilBertTokenizer
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model.save_pretrained("models/nlp/distilbert")
tokenizer.save_pretrained("models/nlp/distilbert")