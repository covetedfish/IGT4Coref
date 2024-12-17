from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("/projects/enri8153/tmp/es_last_model_chrf_big_bf16")
import transformers
import pandas as pd
import transformers
import datasets
from transformers import T5ForConditionalGeneration
import numpy as np
from datasets import load_dataset

dataset = load_dataset('csv', data_files={"train": 'CorefUD_Spanish-AnCora/es_ancora-corefud-train.csv', "val": 'CorefUD_Spanish-AnCora/es_ancora-corefud-dev.csv'})

tokenizer = transformers.ByT5Tokenizer.from_pretrained(
        "google/byt5-small", use_fast=False
    )


def tokenize(examples):
    # Tokenize both input and target sequences
    inputs = tokenizer(
      examples['Input_Block'],
      text_target=examples['Output_Block'],
      truncation=True,
      padding=False,
      max_length=500
    )
    return inputs


dataset = dataset.map(tokenize, batched=True)

training_args =  transformers.Seq2SeqTrainingArguments(
        output_dir="./tmp_chrf",
        max_steps=1000,
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=96,
        generation_num_beams=1,
        metric_for_best_model="chrf++",
        dataloader_num_workers=1,
        bf16=True,
        generation_max_length=500
    )

# metric = load_metric('accuracy')

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels)

trainer = transformers.Seq2SeqTrainer(
        model=model, 
        args=training_args,  # Training arguments, defined above
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        )
    )

preds = trainer.predict(dataset['val'])  # type:ignore
print(preds)
labels = np.where(
            preds.predictions != -100, preds.predictions, tokenizer.pad_token_id
        )
preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
preds_path = "./big_chrf_model.csv"
preds_df = pd.DataFrame(
            {
                "pred": preds,
                "gold": dataset['val']['Output_Block'],
            }
        )
preds_df.to_csv(preds_path, index=False)

