
import logging
import os
import random
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from argparse import ArgumentParser
import nltk
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(0)
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

# Initialize logging
logging.basicConfig(
    filename='logs/log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded_dict['input_ids'].flatten(), encoded_dict['attention_mask'].flatten(), torch.tensor(label,
                                                                                                           dtype=torch.float)


class DataAugmentation:
    def __init__(self):
        pass

    def synonym_replace(self, text):
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = wordnet.synsets(random_word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                if synonym != random_word:
                    new_words = [synonym if word == random_word else word for word in new_words]
                    num_replaced += 1
            if num_replaced >= 1:  # You can increase this number for more replacements
                break
        return ' '.join(new_words)


def get_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in data_loader:
        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        all_preds.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    mse = (np.square(all_labels - all_preds)).mean()
    pearson_r = stats.pearsonr(all_preds, all_labels)[0]

    # Log predicted and true labels for debugging
    #logging.info(f"Predicted labels: {all_preds}")
    #logging.info(f"True labels: {all_labels}")

    # Assuming binary classification for accuracy and F1 score calculations
    predicted_labels = (all_preds > 0.5).astype(int)
    true_labels = (all_labels > 0.5).astype(int)
    #acc = accuracy_score(true_labels, predicted_labels)
    #f1 = f1_score(true_labels, predicted_labels)

    # Check for presence of positive samples in both predicted and true labels
    if np.sum(predicted_labels) == 0 or np.sum(true_labels) == 0:
        logging.warning("No positive samples in either predicted or true labels. Adjusting zero_division parameter.")
        acc = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, zero_division=1)
    else:
        acc = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

    return mse, pearson_r, acc, f1


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(data_loader):
        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def arguments():
    parser = ArgumentParser()
    parser.set_defaults(show_path=False, show_similarity=False)

    parser.add_argument('--mode')
    parser.add_argument('--pre_trained_model_name_or_path', default='bert-base-uncased')
    parser.add_argument('--train_path', default='train.csv')
    parser.add_argument('--val_path', default='val.csv')
    parser.add_argument('--test_path', default='test.csv')
    parser.add_argument('--log_saving_path', default='log.log')
    parser.add_argument('--predict_data_path')
    parser.add_argument('--model_saving_path', default=None)
    parser.add_argument('--test_saving_path', default=None)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cuda', action='store_true', help="Use CUDA if available")

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    data_aug = DataAugmentation()


    def load_data(path, tokenizer, augment=False):
        df = pd.read_csv(path)
        texts = df.iloc[:, 0].tolist()
        labels = df.iloc[:, -1].astype(float).tolist()
        if augment:
            augmented_texts = []
            augmented_labels = []
            for text, label in zip(texts, labels):
                augmented_texts.append(text)
                augmented_labels.append(label)
                augmented_texts.append(data_aug.synonym_replace(text))
                augmented_labels.append(label)
            texts = augmented_texts
            labels = augmented_labels
        return TextDataset(texts, labels, tokenizer, max_len=args.max_len)


    tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.pre_trained_model_name_or_path, num_labels=1)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.mode == 'train':
        logging.info(f"Starting training with model: {args.pre_trained_model_name_or_path}")
        train_dataset = load_data(args.train_path, tokenizer, augment=True)
        val_dataset = load_data(args.val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        best_r = -1
        best_acc = -1
        best_f1 = -1

        for epoch in range(args.max_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_pearson, val_acc, val_f1 = get_metrics(model, val_loader, device)
            logging.info(
                f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val Pearson = {val_pearson}, Val Acc = {val_acc}, Val F1 = {val_f1}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss, best_r, best_acc, best_f1 = val_loss, val_pearson, val_acc, val_f1
                if args.model_saving_path:
                    model.save_pretrained(args.model_saving_path)
                    tokenizer.save_pretrained(args.model_saving_path)

        logging.info(f"Best validation loss: {best_val_loss}")
        logging.info(f"Best test loss: {best_test_loss}")
        logging.info(f"Best test Pearson correlation: {best_r}")
        logging.info(f"Best test accuracy: {best_acc}")
        logging.info(f"Best test F1 score: {best_f1}")
        logging.info(f"Model saved at {args.model_saving_path}/{args.pre_trained_model_name_or_path}")

    elif args.mode == 'predict':
        test_dataset = load_data(args.predict_data_path, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        test_preds, test_labels = get_metrics(model, test_loader, device)

        df = pd.read_csv(args.predict_data_path)
        df['score'] = test_preds
        df.to_csv(args.test_saving_path, index=False)
