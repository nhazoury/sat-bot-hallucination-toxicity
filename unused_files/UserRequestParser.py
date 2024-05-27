from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import pandas as pd
import openai

id2label = {0: "chat", 1: "news", 2: "dive in", 3: "search"}
label2id = {"chat": 0, "news": 1, "dive in": 2, "search": 3}

class CustomDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer: AutoTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        input = self.dataset["data"].values[idx]
        label = self.dataset["label"].values[idx]
        return (input, label)
    
    def pad_data(self, data):
        inputs = [x[0] for x in data]
        labels = [x[1] for x in data]
        inputs_encoding = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
        if isinstance(labels[0], str):
            labels_encoding = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True, max_length=512)
        else:
            labels_encoding = torch.LongTensor(labels)

        return inputs_encoding, labels_encoding, inputs, labels

    def collate_fn(self, all_data):

        inputs_encoding, labels_encoding, inputs, labels = self.pad_data(all_data)
        batched_data = {
                'inputs_encoding': inputs_encoding, 
                'labels_encoding': labels_encoding,
                "inputs": inputs,
                "labels": labels
            }

        return batched_data

class DistilbertParser:
    def __init__(self, model_name="./distilbert-finetuned", save_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                        num_labels=4, id2label=id2label, label2id=label2id)
        self.model.eval()
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        if self.use_cuda:
            self.model = self.model.cuda()

    def train(self, train_df: pd.DataFrame, batch_size=5, num_epochs=100, lr=1e-5, seed=0):
        '''
            Train the underlying model, using the data from train_df, with 2 columns: data (str) and label (integer 0,1,2)

            Optimizer is dedault to AdamW
            loss is default to Cross Entropy
        '''
        torch.manual_seed(seed)
        random.seed(seed)
        train_dataset = CustomDataset(train_df, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate_fn)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.model.train()
        for epoch in range(num_epochs):
            train_loss = 0.0
            num_batches = 0  
            for idx, batch in enumerate(train_dataloader):
                inputs_encoding, labels_encoding, inputs, labels = batch["inputs_encoding"], batch["labels_encoding"], batch["inputs"], batch["labels"]
                input_ids = inputs_encoding["input_ids"]
                
                if self.use_cuda:
                    input_ids = input_ids.to("cuda")
                    labels_encoding = labels_encoding.to("cuda")

                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                loss = loss_fn(probs, labels_encoding.view(-1)) / len(inputs_encoding)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
            train_loss = train_loss / (num_batches)
            print(f"epoch: {epoch}, epoch loss :: {train_loss}")
        if self.save_path:
            self.model.save_pretrained(self.save_path)
        return self.model
    
    def inference(self, input: str):
        '''
            Given an string input (user query), predict what next step we should take

            Returns:
                0 for do nothing, 1 for search, 2 for dive in
        '''
        input_ids = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True).input_ids
        if self.use_cuda:
            input_ids = input_ids.to("cuda")
        
        logits = self.model(input_ids).logits
        logits = logits.detach().cpu().numpy()  
        preds = np.argmax(logits, axis=1).flatten()
        
        return preds[0]

def gpt_classify(text, timeout: float=3):
    all_messages =  [{"role": "system", "content": "given a query reply \"chat\" if it's just casual conversation. Write \"query\" if it's a question that doesn't require information after 2021. Reply \"search\" if it's an information question something more recent like \"who is the president?\" or \"tell me about gpt-4\".  Finally reply \"news\" if it's news related such as \"What going on in Ukraine?\" or \"What's new?\""}]
    all_messages += [{"role": "user", "content": "what's up?"}]
    all_messages += [{"role": "assistant", "content": "chat"}]
    all_messages += [{"role": "user", "content": "who discovered the grand canyon?"}]
    all_messages += [{"role": "assistant", "content": "query"}]
    all_messages += [{"role": "user", "content": "what's the weather like in new york?"}]
    all_messages += [{"role": "assistant", "content": "search"}]
    all_messages += [{"role": "user", "content": "I think so"}]
    all_messages += [{"role": "assistant", "content": "chat"}]
    all_messages += [{"role": "user", "content": text}]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=all_messages,
        request_timeout=timeout,
    )

    result = response.choices[0]['message']['content'].strip()
    return result

if __name__ == "__main__":
    query_parser = DistilbertParser(model_name='distilbert-base-uncased', save_path='./model/distilbert-finetuned')
    train_df = pd.read_excel("data/binary_classifer_text.xlsx", sheet_name="train")
    query_parser.train(train_df, num_epochs=300, batch_size=20)
    query_parser = DistilbertParser(model_name="model/distilbert-finetuned")
    for q in ["What's new?", "What's new in quantum computing?", "What's the weather today?", 
              "Tell me a joke", "Tell me more about it",
              "Any good restaurants nearby?", "Any good place for fun?",
              "Can you tell me more?", "Can you recommend some place to eat?"]:
        prediction = query_parser.inference(q)
        print(f"{q}: {id2label[prediction]}")