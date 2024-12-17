import transformers
import datasets
from transformers import T5ForConditionalGeneration
import numpy as np
from datasets import load_dataset
import torch
from datasets import load_metric
from config_to_dataclass import config_to_dataclass
from dataclasses import dataclass
import argparse 
import os

@dataclass
class ExperimentConfig:

    # General
    exp_name: str = "spanish_small"

    pretrained_model: str = "google/byt5-small"
    language: str = "Spanish"
    # Dataset
    max_tokens: int = 500
    train_path: str = "CorefUD_Spanish-AnCora/es_ancora-corefud-train.csv"
    val_path: str = "CorefUD_Spanish-AnCora/es_ancora-corefud-dev.csv"
    data_dir: str = "data"
    # Training
    metric_for_best_model: str = "chrf++"

    # Files
    predict_path: str = "predictions/"

    model_dir: str = "models/"
    checkpoint_dir: str = "training_checkpoints/"


def _make_if_needed(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run(config: ExperimentConfig):
    output_dir = _make_if_needed(os.path.join(config.checkpoint_dir, config.exp_name))

    dataset = load_dataset(
        'csv',
        data_files={
            "train": f"{config.data_dir}/{config.train_path}",
            "val":  f"{config.data_dir}/{config.val_path}",
        },
    )
    dataset['train'] = dataset['train'] 
    dataset['val'] = dataset['val'].select(range(50))      # First 100 rows
    dataset['test'] = dataset['val'].select(range(50,))      # First 100 rows

    tokenizer = transformers.ByT5Tokenizer.from_pretrained(
            "google/byt5-small", use_fast=False
        )


    def tokenize(examples):
        # Tokenize both input and target sequences
        inputs = tokenizer(
        examples['Input_Block'],
        text_target=examples['Output_Block'],
        truncation=True,
        padding=True,
        max_length=config.max_tokens)
        return inputs

    torch.backends.cudnn.benchmark = True

    dataset = dataset.map(tokenize, batched=True)

    predict_with_generate = False
    if config.metric_for_best_model == "chrf++":
         predict_with_generate = True

    model = T5ForConditionalGeneration.from_pretrained("lecslab/glosslm")
    model.gradient_checkpointing_enable()
    model = model.to("cuda")
    training_args =  transformers.Seq2SeqTrainingArguments(
            output_dir=output_dir,
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
            predict_with_generate= predict_with_generate,
            logging_steps=96,
            generation_num_beams=1,
            metric_for_best_model= config.metric_for_best_model,
            dataloader_num_workers=1,
            bf16=True,
            generation_max_length=config.max_tokens
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
    trainer.train()
    trainer.save_model("f{config.model_dir}/{config.language}_{config.metric_for_best_model}_{config.max_tokens}")

    preds = trainer.predict(dataset['test'])  # type:ignore
    labels = np.where(
                preds.predictions != -100, preds.predictions, tokenizer.pad_token_id
            )
    preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds_path = "f{config.predict_dir}/{config.language}_{config.metric_for_best_model}_{config.max_tokens}"
    preds_df = pd.DataFrame(
            {
                "pred": preds,
                "gold": dataset['val']['Output_Block'],
            }
        )
    preds_df.to_csv(preds_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "-o",
        "--overrides",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or "",
        dataclass_type=ExperimentConfig,
    )
    run(config)
