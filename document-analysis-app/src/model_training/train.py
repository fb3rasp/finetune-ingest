# import os
# import json
# import torch
# from torch.utils.data import DataLoader
# from transformers import LlamaForCausalLM, LlamaTokenizer

# class DocumentDataset(torch.utils.data.Dataset):
#     def __init__(self, data_file):
#         self.data = self.load_data(data_file)

#     def load_data(self, data_file):
#         with open(data_file, 'r') as f:
#             return json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# def train_model(data_file, model_name, output_dir, epochs=3, batch_size=8, learning_rate=5e-5):
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     model = LlamaForCausalLM.from_pretrained(model_name)

#     dataset = DocumentDataset(data_file)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     model.train()
#     for epoch in range(epochs):
#         for batch in dataloader:
#             inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
#             outputs = model(**inputs, labels=inputs['input_ids'])
#             loss = outputs.loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)

# if __name__ == "__main__":
#     data_file = os.path.join('data', 'training_data.json')  # Adjust path as necessary
#     model_name = 'Llama-3.x'  # Replace with the actual model name
#     output_dir = './model_output'
#     train_model(data_file, model_name, output_dir)