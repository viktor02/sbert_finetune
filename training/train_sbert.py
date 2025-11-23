import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 1. КОНФИГУРАЦИЯ
MODEL_NAME = "ai-forever/sbert_large_mt_nlu_ru"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3

# 2. ЗАГРУЗКА ДАННЫХ
df = pd.read_csv("dataset_chunked_train.csv", on_bad_lines="skip")

# Очистка данных (удаление строк с текстом в label)
df = df[pd.to_numeric(df["label"], errors="coerce").notnull()]
df["label"] = df["label"].astype(int)

# Разделение на train и test
# random_state=42 гарантирует, что при каждом запуске в train/test попадут одни и те же строки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- НОВОЕ: Сохраняем разделенные данные, чтобы видеть их глазами ---
print(f"Всего строк: {len(df)}")
print(f"Train строк: {len(train_df)}")
print(f"Test строк:  {len(test_df)}")

train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)
print("Файлы 'train_split.csv' и 'test_split.csv' сохранены.")
# ------------------------------------------------------------------

# Сброс индексов
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. ТОКЕНИЗАЦИЯ
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. ЗАГРУЗКА МОДЕЛИ
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# 5. МЕТРИКИ
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# 6. ПАРАМЕТРЫ ОБУЧЕНИЯ
training_args = TrainingArguments(
    output_dir="./sbert_classifier_result",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    remove_unused_columns=True,
)

# 7. ЗАПУСК TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Начинаем обучение...")
trainer.train()

trainer.save_model("./final_ai_detector")
print("Модель сохранена в папку ./final_ai_detector")
