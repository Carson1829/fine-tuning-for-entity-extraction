import shutil
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import config
from data import get_examples, load_and_group


def get_tokenizer():
    """Load and configure the tokenizer for the base model.
    Sets pad token to eos token and right-pads to avoid attention mask issues."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    """Load the base causal LM in bfloat16 precision.
    device_map='auto' distributes across available GPUs automatically."""
    return AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )


def load_lora_model():
    """Load the base model with the trained LoRA adapter applied.
    Adapter weights are loaded from OUTPUT_DIR saved during training."""
    base  = load_base_model()
    model = PeftModel.from_pretrained(base, config.OUTPUT_DIR)
    model.eval()
    return model


def train_model():
    """Fine-tune the base model using LoRA and save the adapter weights.

    Steps:
    1. Clear any existing adapter in OUTPUT_DIR to avoid loading stale weights
    2. Load base model and tokenizer
    3. Prepare chunked training and validation examples from annotation JSON files
    4. Attach LoRA adapter to the 4 attention projection matrices
    5. Train with HuggingFace Trainer, saving the best checkpoint by validation loss
    6. Save adapter weights and tokenizer to OUTPUT_DIR
    """
    # Clear old adapter to avoid accidentally loading stale weights
    if os.path.exists(config.OUTPUT_DIR):
        shutil.rmtree(config.OUTPUT_DIR)
        print("Cleared old adapter.")

    model     = load_base_model()
    tokenizer = get_tokenizer()

    # Build tokenized training examples from chunked annotated documents
    train_examples = get_examples(load_and_group("train.json"), tokenizer)
    val_examples   = get_examples(load_and_group("val.json"),   tokenizer)

    # Configure LoRA: inject low-rank matrices into all 4 attention projections
    # r=32 and alpha=64 (2x rank) is a standard scaling for LoRA updates
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,  # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
        num_train_epochs=config.TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",        # smooth LR decay after warmup
        warmup_ratio=config.WARMUP_RATIO,  # prevents large gradient updates at start
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,       # saves best checkpoint by val loss, not last epoch
        bf16=True                         
    )

    # DataCollatorForSeq2Seq pads sequences in each batch and sets padding
    # token labels to -100 so loss is not computed on padding tokens
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        )
    )

    trainer.train()

    # Save only the LoRA adapter weights, not the full model
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("Training done.")