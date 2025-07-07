from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
from utils.log_to_csv import *
from utils.log_to_txt import *

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, model_name="VanillaRNN", attention_name="none"):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        log_line = f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}"
        print(log_line)
        log_to_txt(log_line, model_name, attention_name)

    metrics = evaluate_model(model, val_loader, criterion, device)

    log_line = f"Final Val -> Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    print(log_line)

    log_to_txt(log_line, model_name, attention_name)
    log_results(model_name, attention_name, metrics)
    print("\n")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs, _ = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Val Loss: {val_loss / len(data_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print("Confusion Matrix:\n", cm)

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "loss": val_loss / len(data_loader)}
