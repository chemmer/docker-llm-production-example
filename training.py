import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_llama_classifier(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    dataset_name="imdb",  # Example dataset, replace with your own
    output_dir="./llama2_classifier_output",
    num_labels=2,
    batch_size=2,
    learning_rate=2e-5,
    num_epochs=3,
):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 16,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": True,
        },
    }
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    torch.cuda.empty_cache()
    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train_llama_classifier()
