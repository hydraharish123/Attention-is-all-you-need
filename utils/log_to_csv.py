import csv, os

def log_results(model_name, attention_name, metrics, file_path="output/metrics_log.csv"):
    ## check if the output file is there or not 
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)

    ## for each base model + attention, we save the evaluation metrics to later compare the models
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Model", "Attention", "Accuracy", "F1", "Precision", "Recall", "Loss"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Model": model_name, "Attention": attention_name, "Accuracy": round(metrics['accuracy'], 4), "F1": round(metrics['f1'], 4), "Precision": round(metrics['precision'], 4), "Recall": round(metrics['recall'], 4), "Loss": round(metrics['loss'], 4)})
