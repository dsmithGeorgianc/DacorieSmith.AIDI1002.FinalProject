# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
#
#
# def load_model_and_tokenizer(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     if torch.cuda.is_available():
#         model.cuda()
#     return tokenizer, model
#
#
# def encode_batch(text, tokenizer, max_len=50):
#     encoded_dict = tokenizer.batch_encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=max_len,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
#     return encoded_dict['input_ids'], encoded_dict['attention_mask']
#
#
# def predict(model, tokenizer, texts, max_len=50):
#     model.eval()
#     input_ids, attention_masks = encode_batch(texts, tokenizer, max_len)
#
#     if torch.cuda.is_available():
#         input_ids = input_ids.cuda()
#         attention_masks = attention_masks.cuda()
#
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_masks)
#         predictions = outputs[0]
#
#     return predictions.cpu().numpy()
#
#
# # Paths to models and example texts
# model_paths = {
#    # "bert-base-uncased": "/Users/dacoriesmith/PycharmProjects/business_uccession_analytics_planning/machine_learning_programming/quantifying-stereotypes-in-language/models/bert-base-uncased/bert-base-uncased",
#     "distilbert-base-uncased": "/Users/dacoriesmith/PycharmProjects/business_uccession_analytics_planning/machine_learning_programming/quantifying-stereotypes-in-language/models/distilbert-base-uncased/distilbert-base-uncased",
#     "roberta-base": "/Users/dacoriesmith/PycharmProjects/business_uccession_analytics_planning/machine_learning_programming/quantifying-stereotypes-in-language/models/roberta-base/roberta-base"
# }
#
# texts = ["Example sentence 1 black people suck", "Example sentence 2 white people goood"]
#
# for model_name, model_path in model_paths.items():
#     print(f"Using model: {model_name}")
#     tokenizer, model = load_model_and_tokenizer(model_path)
#     predictions = predict(model, tokenizer, texts)
#
#     for text, prediction in zip(texts, predictions):
#         print(f"Text: {text}\nPrediction: {prediction}\n")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model


def encode_batch(text, tokenizer, max_len=50):
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def predict(model, tokenizer, texts, max_len=50):
    model.eval()
    input_ids, attention_masks = encode_batch(texts, tokenizer, max_len)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_masks = attention_masks.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        predictions = outputs[0]

    return predictions.cpu().numpy()


def categorize_prediction(prediction):
    if prediction < 0.01:
        return 'Not Offensive'
    elif 0.01 <= prediction <= 0.10:
        return 'Not So Bad'
    else:
        return 'Super Bad'


# Base path to models
base_path = "/Users/dacoriesmith/PycharmProjects/business_uccession_analytics_planning/machine_learning_programming/quantifying-stereotypes-in-language/models/"

# Paths to models with the base path
model_paths = {
    "distilbert-base-uncased": f"{base_path}distilbert-base-uncased/distilbert-base-uncased",
    "roberta-base": f"{base_path}roberta-base/roberta-base",
    "bert-base-uncased": f"{base_path}bert-base-uncased",
    "bert-base-cased": f"{base_path}bert-base-cased/bert-base-cased",
    "distilbert-base-cased": f"{base_path}distilbert-base-cased/distilbert-base-cased",
    "xlnet-base-cased": f"{base_path}xlnet-base-cased/xlnet-base-cased",
    "albert-base-v2": f"{base_path}albert-base-v2",
    "google/electra-small-discriminator": f"{base_path}google_electra-small-discriminator",
    "microsoft/deberta-base": f"{base_path}microsoft_deberta-base",
    "t5-small": f"{base_path}t5-small"
}

texts_with_categories_ratings = [
    ("Black people are too poor to drive good cars.", "race", 0.3108291546889673),
    ("In bangladesh they work in a factory.", "race", -0.1472278527656165),
    ("The corrupt politician won the majority of the votes.", "profession", 0.1328436568751042),
    ("The stupid people are russian.", "race", 0.2399742854223964),
    (
    "I dated a nerd, and was very happy, he always had this schoolboy attitude, we had lots of fun together.", "gender",
    -0.3789638351049204),
    ("Many refugees came from syria into the united states.", "race", -0.2335353928033051),
    ("The people of cape verde have a very distinguished air about them.", "race", -0.2584535600863498),
    ("As a psychologist, Dr. B. is very observant of people's behaviors.", "profession", -0.344071009337297),
    ("Lisa ran across the street, tackled the dog to save Beth and her dog from attack.", "gender", -0.161694354709108),
    ("Christians follow the word of God.", "religion", -0.0817439415560607),
    (
    "The race had begun, but the Mexican runner was just jogging while everyone else was running, god they are so lazy.",
    "race", 0.3061076511466941),
]

results = []

for model_name, model_path in model_paths.items():
    print(f"Using model: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_path)
    texts = [text for text, category, rating in texts_with_categories_ratings]
    predictions = predict(model, tokenizer, texts)

    for (text, category, rating), prediction in zip(texts_with_categories_ratings, predictions):
        category_result = categorize_prediction(prediction.max())
        print(
            f"Text: {text}\nCategory: {category}\nRating: {rating}\nPrediction: {prediction}\nCategory Result: {category_result}\n")
        results.append(
            {'Model': model_name, 'Text': text, 'Category': category, 'Rating': rating, 'Prediction': prediction.max(),
             'Category Result': category_result})

# Convert results to DataFrame for easier analysis
df_results = pd.DataFrame(results)
print(df_results)

# Aggregate the results for plotting
aggregate_results = df_results.groupby(['Model', 'Category Result']).size().unstack(fill_value=0)

# Plotting the results
bar_width = 0.1  # Adjusted bar width for better spacing
index = np.arange(len(aggregate_results.columns))

fig, ax = plt.subplots(figsize=(14, 8))  # Increased figure size for better clarity

# Loop through each model to plot the bars with adjusted bar width and consistent color mapping
colors = plt.cm.get_cmap('tab10', len(aggregate_results.index))  # Use a colormap for consistent colors

for i, (model, color) in enumerate(zip(aggregate_results.index, colors.colors)):
    ax.bar(index + i * bar_width, aggregate_results.loc[model], bar_width, label=model, color=color)

# Adding gridlines
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Setting labels and title
plt.xlabel('Category Result', fontsize=12)
plt.ylabel('Number of Texts', fontsize=12)
plt.title('Model Prediction Categorization', fontsize=14)

# Adjusting x-ticks and legend
plt.xticks(index + bar_width * (len(aggregate_results.index) - 1) / 2, aggregate_results.columns)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Models")  # Legend outside the plot for better visibility

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()