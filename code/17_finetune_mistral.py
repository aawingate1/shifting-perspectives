#!/usr/bin/env python3
"""
Mistral Fine-Tuning Script

Fine-tunes Mistral-7B-Instruct on BBQ training data for bias mitigation.
Uses preference-based training with positive/negative example pairs.
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Logger will be configured in main()
logger = logging.getLogger(__name__)



def load_bbq_training_data():
    """Load and randomly sample 300 BBQ training examples from JSON files."""
    
    import random
    
    bbq_train_dir = Path("../data/bbq_train")
    all_data = []
    
    # All bias categories
    categories = ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']
    
    # Set random seed for reproducibility
    random.seed(42)
    
    for category in categories:
        json_file = bbq_train_dir / f"{category}_train.json"
        
        if not json_file.exists():
            logger.warning(f"Training file not found: {json_file}")
            continue
            
        logger.info(f"Loading {category} training data...")
        
        with open(json_file, 'r') as f:
            category_data = json.load(f)
            
        # Add category label and process data
        for item in category_data:
            # Use positive examples for training (bias-mitigated responses)
            all_data.append({
                'text': item['positive'],
                'category': category,
                'type': 'positive'
            })
            
        logger.info(f"Loaded {len(category_data)} examples from {category}")
    
    logger.info(f"Total available training examples: {len(all_data)}")
    
    # Randomly sample 300 examples
    n_samples = 300
    if len(all_data) > n_samples:
        sampled_data = random.sample(all_data, n_samples)
        logger.info(f"Randomly sampled {n_samples} examples from {len(all_data)} total")
    else:
        sampled_data = all_data
        logger.info(f"Using all {len(all_data)} examples (less than requested {n_samples})")
    
    # Log distribution by category
    category_counts = {}
    for item in sampled_data:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    logger.info("Sample distribution by category:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count} examples")
    
    return sampled_data


def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess the training data for fine-tuning."""
    
    # Tokenize the text
    model_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Add padding
        return_tensors=None
    )
    
    # For causal language modeling, labels are the same as input_ids
    # Make sure labels are also padded to max_length
    model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
    
    return model_inputs


def setup_model_and_tokenizer(model_name):
    """Set up the model and tokenizer for fine-tuning."""
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )
    
    # Set up LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,  # Scaling parameter
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main fine-tuning function."""
    
    # Configuration
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    output_dir = "../models/mistral-bbq-finetuned"
    
    # Set HuggingFace cache directory
    os.environ["HF_HOME"] = "/scratch/zara/.cache/huggingface"
    
    # Set up logging to file in logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/mistral_finetune_{timestamp}.log"
    os.makedirs("../logs", exist_ok=True)
    
    # Configure logging to write to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting Mistral fine-tuning on BBQ data...")
    logger.info(f"Logging to: {log_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    logger.info("Loading BBQ training data...")
    training_data = load_bbq_training_data()
    
    if not training_data:
        logger.error("No training data found!")
        return
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(training_data)
    logger.info(f"Created dataset with {len(dataset)} examples")
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
        pad_to_multiple_of=None  # Disable padding since we already pad in preprocessing
    )
    
    # Training arguments    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb/tensorboard
        run_name=f"mistral-bbq-finetune-{timestamp}"
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    
    try:
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            'model_name': model_name,
            'training_examples': len(training_data),
            'epochs': training_args.num_train_epochs,
            'learning_rate': training_args.learning_rate,
            'timestamp': timestamp,
            'output_dir': output_dir
        }
        
        info_file = Path(output_dir) / "training_info.json"
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("="*60)
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training examples: {len(training_data)}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()