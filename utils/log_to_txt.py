import os 

def log_to_txt(log_str, model_name, attention_name, file_path="output/training_logs.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"[{model_name} | {attention_name}] {log_str}\n")