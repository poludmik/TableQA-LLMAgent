from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
sequence_to_classify = "Pie chart 3 largest gdps. Add values. Use red color for the biggest one. Add shadow. Title it 'GDP'."
# sequence_to_classify = "draw me the speed compared to charging"
# sequence_to_classify = "What is the maximum speed?"
# sequence_to_classify = "What are the 3 best accelerations?"


model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

hypothesis = "A plot, a chart, a visualization, or a graph"

acc = 0
acc_plot = 0
df = pd.read_excel("dataset/dataset_250.xlsx")
# iterate over rows of a dataframe "dataset/dataset_250.xlsx"
for index, row in df.iterrows():
    # predict the label for the user query
    sequence_to_classify = row["user_query"]
    # print(f"\nIndex: {index}, sequence_to_classify: {sequence_to_classify}")
    if sequence_to_classify == "" or str(sequence_to_classify) == "nan":  # for the last row with plot counts
        continue
    input = tokenizer(sequence_to_classify, hypothesis, truncation=False, return_tensors="pt")
    output = model(input["input_ids"].to("cpu"))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["true", "neutral"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    # print(f"Index: {index}, prediction: {prediction}")

    prediction_plot = prediction["true"]
    prediction_general = prediction["neutral"]
    if prediction_plot > prediction_general:
        label = 1
    else:
        label = 0

    if label == row["has_plot_answer"]:
        acc += 1
        if row["has_plot_answer"] == 1:
            acc_plot += 1

print("hyp:", hypothesis)
print(f"Accuracy: {acc / len(df)}")
print(f"Accuracy for plot: {acc_plot / 65}")

