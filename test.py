
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

models = [
    "bert-base-uncased",
      "bert-base-cased",
      "distilbert-base-uncased",
       "distilbert-base-cased",
       "roberta-base",
      "xlnet-base-cased",
         "albert-base-v2",
      "google/electra-small-discriminator",
       "microsoft/deberta-base",
     "t5-small",
    "gpt2",
]

base_command = "python /Users/dacoriesmith/PycharmProjects/business_uccession_analytics_planning/machine_learning_programming/quantifying-stereotypes-in-language/train.py --mode train"

train_path = "data/train.csv"
val_path = "data/val.csv"
test_path = "data/test.csv"
lr = "0.00001"
max_len = "50"
max_epochs = "30"
batch_size = "128"
model_saving_path = "models"

def train_model(model):
    model_path = f"{model_saving_path}/{model.replace('/', '_')}"
    if not os.path.exists(model_path):
        command = (
            f"{base_command} "
            f"--pre_trained_model_name_or_path {model} "
            f"--train_path {train_path} "
            f"--val_path {val_path} "
            f"--test_path {test_path} "
            f"--lr {lr} "
            f"--max_len {max_len} "
            f"--max_epochs {max_epochs} "
            f"--batch_size {batch_size} "
            f"--model_saving_path {model_path}"
        )
        print(f"Training model: {model}")
        print(command)
        os.system(command)
    else:
        print(f"Model already exists: {model}")

def main():
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_model, model) for model in models]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()
