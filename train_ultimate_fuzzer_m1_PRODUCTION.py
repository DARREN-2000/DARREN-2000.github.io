#!/usr/bin/env python3
"""
Ultimate Fuzzer M1 Production Training Script
Enhanced ML model for automotive security testing with LLM augmentation.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SecurityTestDataset(Dataset):
    """Custom dataset for security test cases and vulnerability patterns."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FuzzerModel(nn.Module):
    """Enhanced model architecture for security testing prediction."""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super(FuzzerModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

class ProgressCallback(TrainerCallback):
    """Custom callback for real-time training progress monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("🚀 Training started - Ultimate Fuzzer M1 Production")
        logger.info("=" * 80)
    
    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_log_time >= 10:  # Log every 10 seconds
            elapsed = current_time - self.start_time
            progress = state.global_step / state.max_steps if state.max_steps else 0
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            
            logger.info(f"Step {state.global_step:>6} | "
                       f"Progress: {progress*100:>6.2f}% | "
                       f"Elapsed: {elapsed:>8.1f}s | "
                       f"ETA: {eta:>8.1f}s")
            self.last_log_time = current_time
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items() 
                                    if isinstance(v, (int, float))])
            logger.info(f"📊 Metrics: {metrics_str}")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info(f"✅ Training completed in {total_time:.2f} seconds")

class MetricsCallback(TrainerCallback):
    """Advanced metrics tracking and logging."""
    
    def __init__(self):
        self.best_metrics = {}
        self.epoch_metrics = []
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Track best metrics
            for key, value in logs.items():
                if 'eval_' in key and isinstance(value, (int, float)):
                    metric_name = key.replace('eval_', '')
                    if (metric_name not in self.best_metrics or 
                        (metric_name in ['accuracy', 'f1'] and value > self.best_metrics[metric_name]) or
                        (metric_name in ['loss'] and value < self.best_metrics[metric_name])):
                        self.best_metrics[metric_name] = value
                        logger.info(f"🎯 New best {metric_name}: {value:.4f}")
            
            # Log current epoch metrics
            self.epoch_metrics.append({
                'epoch': state.epoch,
                'step': state.global_step,
                **logs
            })

def load_and_prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[List, List, List, List]:
    """Load and prepare security testing data."""
    logger.info(f"Loading data from {data_path}")
    
    # Simulate loading real automotive security data
    # In production, this would load from actual vulnerability databases
    security_patterns = [
        "SQL injection attempt in authentication module",
        "Buffer overflow detected in CAN bus communication",
        "Cross-site scripting in vehicle dashboard interface",
        "Memory corruption in ECU firmware validation",
        "Authentication bypass in remote diagnostics",
        "Code injection in OTA update mechanism",
        "Race condition in critical safety systems",
        "Integer overflow in sensor data processing",
        "Path traversal in file system access",
        "Command injection in diagnostic protocols"
    ]
    
    # Generate synthetic training data
    np.random.seed(random_state)
    texts = []
    labels = []
    
    for i in range(10000):
        base_pattern = np.random.choice(security_patterns)
        # Add variations and noise
        variations = [
            f"Detected: {base_pattern}",
            f"Warning: Potential {base_pattern.lower()}",
            f"Alert: {base_pattern} in automotive system",
            f"Security scan found: {base_pattern}",
            f"Vulnerability assessment: {base_pattern}"
        ]
        text = np.random.choice(variations)
        
        # Add contextual information
        contexts = [
            " during penetration testing",
            " in production environment", 
            " via automated scanning",
            " through manual code review",
            " in CI/CD pipeline validation"
        ]
        text += np.random.choice(contexts)
        
        texts.append(text)
        # Binary classification: 0 = safe, 1 = vulnerability
        labels.append(1 if "injection" in base_pattern.lower() or 
                           "overflow" in base_pattern.lower() or
                           "bypass" in base_pattern.lower() else 0)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    logger.info(f"Data loaded: {len(train_texts)} training, {len(test_texts)} testing samples")
    logger.info(f"Label distribution - Train: {np.bincount(train_labels)}, Test: {np.bincount(test_labels)}")
    
    return train_texts, test_texts, train_labels, test_labels

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the model."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def setup_wandb(config: Dict):
    """Initialize Weights & Biases for experiment tracking."""
    wandb.init(
        project="ultimate-fuzzer-m1-production",
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=["production", "automotive", "security", "fuzzing", "llm-enhanced"]
    )
    logger.info("🔗 Weights & Biases initialized for experiment tracking")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Ultimate Fuzzer M1 Production Training")
    parser.add_argument("--model_name", default="microsoft/codebert-base", 
                       help="Pre-trained model name")
    parser.add_argument("--data_path", default="./data/security_dataset.json",
                       help="Path to training data")
    parser.add_argument("--output_dir", default="./models/ultimate_fuzzer_m1_prod",
                       help="Output directory for model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Enable mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("🔧 Initializing Ultimate Fuzzer M1 Production Training")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    
    # Initialize experiment tracking
    if args.use_wandb:
        setup_wandb(vars(args))
    
    # Load and prepare data
    train_texts, test_texts, train_labels, test_labels = load_and_prepare_data(args.data_path)
    
    # Initialize tokenizer and model
    logger.info("🔤 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = FuzzerModel(args.model_name, num_labels=2)
    
    # Create datasets
    train_dataset = SecurityTestDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset = SecurityTestDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure training arguments - duplicate greater_is_better parameter issue fixed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,  # First occurrence
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if args.use_wandb else None,
        run_name=f"ultimate_fuzzer_m1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=args.seed,
        data_seed=args.seed,
        bf16=False,
        tf32=True if torch.cuda.is_available() else False,
        auto_find_batch_size=False,
        gradient_checkpointing=False,
        include_inputs_for_metrics=False,
        prediction_loss_only=False,
        label_smoothing_factor=0.0,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        ddp_timeout=1800,
        skip_memory_metrics=True,
        use_legacy_prediction_loop=False,
        hub_model_id=None,
        hub_strategy="every_save",
        hub_token=None,
        hub_private_repo=False,
        disable_tqdm=False,
        log_level="info",
        log_level_replica="warning",
        log_on_each_node=True,
        logging_nan_inf_filter=True,
        save_safetensors=True,
        ignore_data_skip=False,
        fsdp="",
        fsdp_min_num_params=0,
        fsdp_config=None,
        fsdp_transformer_layer_cls_to_wrap=None,
        accelerator_config=None,
        deepspeed=None,
        label_names=None,
        resume_from_checkpoint=None,
        torch_compile=False,
        torch_compile_backend="inductor",
        torch_compile_mode=None
    )
    
    # Initialize callbacks
    progress_callback = ProgressCallback()
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[progress_callback, metrics_callback, early_stopping]
    )
    
    # Start training
    logger.info("🎯 Starting training process...")
    logger.info("=" * 80)
    
    try:
        train_result = trainer.train()
        
        # Log training results
        logger.info("📈 Training Results:")
        for key, value in train_result.metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Final evaluation
        logger.info("🔍 Running final evaluation...")
        eval_result = trainer.evaluate()
        
        logger.info("📊 Final Evaluation Results:")
        for key, value in eval_result.items():
            logger.info(f"  {key}: {value}")
        
        # Save model and tokenizer
        logger.info(f"💾 Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training configuration
        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Log best metrics
        logger.info("🏆 Best Metrics:")
        for key, value in metrics_callback.best_metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ Training completed successfully!")
        
        if args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        if args.use_wandb:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()