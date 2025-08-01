from google.colab import files
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

class OptimizedConfig:
    """1835 dataset için optimize edilmiş configuration."""

    BASE_MODEL_NAME = "bert-base-uncased"
    MODEL_SAVE_PATH = "/content/crypto_risk_classifier_bert"

    # optimize edilmiş hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 8
    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.2
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    ENHANCE_PATTERNS = True
    USE_CLASS_WEIGHTS = False

    LABEL_NAMES = [
        "Important",
        "Medium",
        "Unimportant"
    ]

    LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
    ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

config = OptimizedConfig()

def get_device():
    """Device selection optimized for 1835 dataset."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")

        if gpu_memory >= 16:
            config.BATCH_SIZE = 24
        elif gpu_memory >= 8:
            config.BATCH_SIZE = 16
        else:
            config.BATCH_SIZE = 12

        print(f"Batch size set to: {config.BATCH_SIZE}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slower)")
        config.BATCH_SIZE = 8

    return device

def enhance_text_with_patterns(text: str, label: str = None) -> str:
    """
    Enhance text with pattern indicators that help model learn
    what makes news Important/Medium/Unimportant
    """
    if not config.ENHANCE_PATTERNS:
        return text

    enhanced_text = text

    financial_patterns = {
        'billions': r'\$?\d+\.?\d*\s*billion|\$?\d+\.?\d*B',
        'millions': r'\$?\d+\.?\d*\s*million|\$?\d+\.?\d*M',
        'thousands': r'\$?\d+\.?\d*\s*thousand|\$?\d+\.?\d*K'
    }

    regulatory_keywords = [
        'SEC', 'regulatory', 'approval', 'lawsuit', 'guilty', 'fine',
        'government', 'legal', 'court', 'ruling', 'banned', 'regulation'
    ]

    market_impact_keywords = [
        'market', 'price', 'surge', 'crash', 'rally', 'dump',
        'all-time high', 'ATH', 'bear market', 'bull market'
    ]

    security_keywords = [
        'hack', 'exploit', 'vulnerability', 'breach', 'stolen',
        'scam', 'fraud', 'attack', 'security'
    ]

    major_entities = [
        'Bitcoin', 'Ethereum', 'Binance', 'Coinbase', 'Tesla',
        'MicroStrategy', 'BlackRock', 'Grayscale', 'PayPal'
    ]

    text_lower = enhanced_text.lower()

    if re.search(financial_patterns['billions'], text_lower):
        enhanced_text = "[FINANCIAL_MAJOR] " + enhanced_text
    elif re.search(financial_patterns['millions'], text_lower):
        enhanced_text = "[FINANCIAL_MEDIUM] " + enhanced_text
    elif re.search(financial_patterns['thousands'], text_lower):
        enhanced_text = "[FINANCIAL_MINOR] " + enhanced_text
    if any(keyword in text_lower for keyword in regulatory_keywords):
        enhanced_text = "[REGULATORY_IMPACT] " + enhanced_text
    if any(keyword in text_lower for keyword in market_impact_keywords):
        enhanced_text = "[MARKET_IMPACT] " + enhanced_text
    if any(keyword in text_lower for keyword in security_keywords):
        enhanced_text = "[SECURITY_RISK] " + enhanced_text
    if any(entity in enhanced_text for entity in major_entities):
        enhanced_text = "[MAJOR_ENTITY] " + enhanced_text

    return enhanced_text

def advanced_clean_text(text: str) -> str:
    """Enhanced text cleaning for crypto news."""
    if not text or pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r'http[s]?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '[USER]', text)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'[HASHTAG] \1', text)
    text = re.sub(r'[^\w\s\[\]$%.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) > 400:
        text = text[:400]

    return text

def upload_and_load_1835_dataset():
    print("CSV dosyanızı yükleyin:")
    uploaded = files.upload()

    if not uploaded:
        print("Dosya yüklenmedi!")
        return None

    filename = list(uploaded.keys())[0]
    print(f"{filename} dosyası yüklendi")

    try:
        encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(filename, encoding=encoding)
                print(f"{encoding} encoding ile başarıyla yüklendi")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print("Hiçbir encoding ile okunamadı")
            return None

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        if 'expected_label' in df.columns:
            df.rename(columns={'expected_label': 'label'}, inplace=True)

        print("Veri temizleniyor...")
        df['text'] = df['text'].astype(str).apply(advanced_clean_text)
        df = df[df['text'].str.strip() != '']
        df['label'] = df['label'].astype(str).str.strip()

        df = df.dropna(subset=['text', 'label'])

        label_standardization = {
            'important': 'Important',
            'IMPORTANT': 'Important',
            'Important': 'Important',
            'medium': 'Medium',
            'MEDIUM': 'Medium',
            'Medium': 'Medium',
            'unimportant': 'Unimportant',
            'UNIMPORTANT': 'Unimportant',
            'Unimportant': 'Unimportant'
        }

        df['label'] = df['label'].map(label_standardization).fillna(df['label'])

        if config.ENHANCE_PATTERNS:
            print("Pattern enhancement uygulanıyor...")
            df['text'] = df.apply(lambda row: enhance_text_with_patterns(row['text'], row['label']), axis=1)

        print(f"\n1835 dataset etiket dağılımı:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

        df['label_id'] = df['label'].map(config.LABEL_TO_ID)

        df = df.dropna(subset=['label_id'])
        df['label_id'] = df['label_id'].astype(int)

        print(f"\n Final dataset: {len(df)} samples")

        return df

    except Exception as e:
        print(f"Dataset yükleme hatası: {e}")
        return None

def create_balanced_datasets(df):
    """Create perfectly balanced train/val/test splits - all classes equal to smallest class."""
    print(f"\n Creating PERFECTLY BALANCED datasets...")

    class_counts = df['label'].value_counts()
    print(f"\n Current class distribution:")
    for label, count in class_counts.items():
        print(f" {label}: {count} samples")

    min_class_size = class_counts.min()
    print(f"All classes will have exactly {min_class_size} samples")

    balanced_samples = []
    for label in config.LABEL_NAMES:
        label_samples = df[df['label'] == label]

        if len(label_samples) >= min_class_size:
            selected_samples = label_samples.sample(n=min_class_size, random_state=42)
            print(f"  {label}: {len(label_samples)} → {min_class_size} samples (downsampled)")
        else:
            selected_samples = label_samples
            print(f"  {label}: {len(label_samples)} samples (all used)")

        balanced_samples.append(selected_samples)

    df_balanced = pd.concat(balanced_samples, ignore_index=True)

    final_counts = df_balanced['label'].value_counts()
    print(f"\nPERFECTLY BALANCED distribution:")
    for label, count in final_counts.items():
        percentage = (count / len(df_balanced)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    total_samples = len(df_balanced)
    samples_per_class = total_samples // 3

    if len(final_counts.unique()) == 1:
        print(f"All classes have exactly {final_counts.iloc[0]} samples")
        print(f"Total balanced dataset: {total_samples} samples ({samples_per_class} per class)")
    else:
        print("Something went wrong with balancing")

    print(f"\nShuffling balanced data...")
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nCreating stratified train/val/test split from balanced data...")

    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=0.30,
        stratify=df_balanced['label_id'],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label_id'],
        random_state=42
    )

    print(f" Balanced split results:")
    print(f"   Training: {len(train_df)} samples ({len(train_df)/len(df_balanced)*100:.1f}%)")
    print(f"   Validation: {len(val_df)} samples ({len(val_df)/len(df_balanced)*100:.1f}%)")
    print(f"   Test: {len(test_df)} samples ({len(test_df)/len(df_balanced)*100:.1f}%)")

    print(f"\n Training set balance:")
    train_counts = train_df['label'].value_counts()
    for label, count in train_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print(f"\n Class weights: DISABLED (perfect balance achieved)")
    class_weights_dict = None

    print(f"\n🔧 Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    print(f" {config.BASE_MODEL_NAME} tokenizer loaded")

    if config.ENHANCE_PATTERNS:
        special_tokens = [
            "[FINANCIAL_MAJOR]", "[FINANCIAL_MEDIUM]", "[FINANCIAL_MINOR]",
            "[REGULATORY_IMPACT]", "[MARKET_IMPACT]", "[SECURITY_RISK]",
            "[MAJOR_ENTITY]", "[URL]", "[USER]", "[HASHTAG]"
        ]

        new_tokens = tokenizer.add_tokens(special_tokens)
        if new_tokens > 0:
            print(f" Added {new_tokens} special pattern tokens")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=512,
            add_special_tokens=True
        )

    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['text', 'label_id']].reset_index(drop=True))

    print("🔧 Tokenizing balanced datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    train_dataset = train_dataset.rename_column('label_id', 'labels')
    val_dataset = val_dataset.rename_column('label_id', 'labels')
    test_dataset = test_dataset.rename_column('label_id', 'labels')

    print("Perfectly balanced datasets created successfully")
    print(f"Final result: {total_samples} samples, {samples_per_class} per class")

    return train_dataset, val_dataset, test_dataset, tokenizer, class_weights_dict

def compute_advanced_metrics(eval_pred):
    """Compute detailed metrics for crypto classification."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    accuracy = (predictions == labels).mean()
    macro_f1 = f1.mean()
    weighted_f1 = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )[2]

    class_accuracy = {}
    for i, label_name in enumerate(config.LABEL_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            class_accuracy[f'accuracy_{label_name}'] = (predictions[mask] == labels[mask]).mean()

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
    }

    metrics.update(class_accuracy)
    return metrics

class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced dataset."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                dtype=torch.float32,
                device=labels.device
            )
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def train_optimized_model():
    """Train model optimized for 1835 crypto news dataset."""
    print("OPTIMIZED TRAINING FOR CRYPTO NEWS DATASET")
    print("=" * 60)

    df = upload_and_load_1835_dataset()
    if df is None:
        return None, None, None

    train_dataset, val_dataset, test_dataset, tokenizer, class_weights = create_balanced_datasets(df)

    print(f"\n Loading model: {config.BASE_MODEL_NAME}")

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config.BASE_MODEL_NAME)
    print(f" {config.BASE_MODEL_NAME} config loaded")

    model_config.num_labels = len(config.LABEL_NAMES)
    model_config.id2label = config.ID_TO_LABEL
    model_config.label2id = config.LABEL_TO_ID
    model_config.hidden_dropout_prob = config.DROPOUT_RATE
    model_config.attention_probs_dropout_prob = config.DROPOUT_RATE

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_NAME,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    print(f" {config.BASE_MODEL_NAME} model loaded")

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="/content/training_output_1835",
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="cosine_with_restarts",
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1",
        greater_is_better=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=42,
        data_seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        compute_metrics=compute_advanced_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    device = get_device()
    print(f"\n OPTIMIZED TRAINING CONFIGURATION:")
    print(f"   Dataset size: 1835 samples")
    print(f"   Model: {config.BASE_MODEL_NAME}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Class weights: {'Enabled' if class_weights else 'Disabled'}")
    print(f"   Pattern enhancement: {'Enabled' if config.ENHANCE_PATTERNS else 'Disabled'}")
    print(f"   Device: {device}")

    print(f"\n Starting optimized training...")
    try:
        trainer.train()
        print(" Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        print(" Try reducing batch size or using a different model")
        return None, None, None

    print(f"\n Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print(f"\n Saving optimized model...")
    trainer.save_model(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)

    print(f"\n" + "=" * 70)
    print(f" OPTIMIZED TRAINING COMPLETED")
    print(f"=" * 70)
    print(f" Model: {config.BASE_MODEL_NAME}")
    print(f" Dataset: 1835 crypto news samples")
    print(f" Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f" Test Macro F1: {test_results['eval_macro_f1']:.4f}")
    print(f" Test Weighted F1: {test_results['eval_weighted_f1']:.4f}")
    print(f" Test Loss: {test_results['eval_loss']:.4f}")

    accuracy = test_results['eval_accuracy']
    if accuracy > 0.85:
        print(f"\n EXCELLENT! Model ready for production")
    elif accuracy > 0.80:
        print(f"\n VERY GOOD! Model performs well")
    elif accuracy > 0.75:
        print(f"\n GOOD! Model is usable")
    else:
        print(f"\n NEEDS IMPROVEMENT! Consider:")

    return trainer, tokenizer, test_dataset

def detailed_evaluation(trainer, tokenizer, test_dataset):
    """Detailed evaluation with crypto-specific analysis."""
    if trainer is None:
        return

    print(f"\n DETAILED EVALUATION")
    print("=" * 50)

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_dataset['labels']

    report = classification_report(y_true, y_pred, target_names=config.LABEL_NAMES, output_dict=True)

    print(f" Class-wise performance:")
    for label in config.LABEL_NAMES:
        if label in report:
            print(f"\n{label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall: {report[label]['recall']:.4f}")
            print(f"  F1-score: {report[label]['f1-score']:.4f}")
            print(f"  Support: {report[label]['support']}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.LABEL_NAMES,
                yticklabels=config.LABEL_NAMES)
    plt.title(f'Confusion Matrix - {config.BASE_MODEL_NAME}\n1835 Crypto News Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    confidence_scores = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
    max_confidences = torch.max(confidence_scores, dim=-1)[0].numpy()

    print(f"\n Confidence Analysis:")
    print(f"  Average confidence: {np.mean(max_confidences):.3f}")
    print(f"  Median confidence: {np.median(max_confidences):.3f}")
    print(f"  Low confidence (<0.6): {np.sum(max_confidences < 0.6)} samples")

def test_with_new_examples(trainer, tokenizer):
    """Test model with new crypto news examples."""
    if trainer is None:
        return

    print(f"\n TESTING WITH NEW EXAMPLES")
    print("=" * 50)

    test_examples = [
        {
            'text': "TRUMP MEDIA BUYS $2 BILLION WORTH OF #BITCOIN AND BITCOIN-RELATED SECURITIES",
            'expected': 'Important'
        },
        {
            'text': "STRATEGY BUYS ANOTHER 6,220 #BITCOIN FOR $739.8 MILLION",
            'expected': 'Important'
        },
        {
            'text': "$763 million #Bitcoin treasury company Nakamoto files with SEC to merge with publicly traded KindlyMD. Expects to complete the merge in 20 days.",
            'expected': 'Unimportant'
        },
        {
            'text': "President Trump is preparing to open the $9 trillion US retirement market to Bitcoin & crypto investments — Financial Times",
            'expected': 'Important'
        },
        {
            'text': "White House says it is exploring de minimis tax exemption for Bitcoin 'to make crypto payments easier.'",
            'expected': 'Important'
        },
        {
            'text': "Congressman Brad Sherman says 'crypto has all the power in Congress.'",
            'expected': 'Unimportant'
        },
        {
            'text': "Thailand to launch a crypto sandbox to allow foreign tourists to use #Bitcoin and crypto in Thailand. 2k",
            'expected': 'Medium'
        },
        {
            'text': "US Marshalls reveal the government now only holds 28,988 Bitcoin worth $3.4 billion, instead of the estimated ~200,000 BTC",
            'expected': 'Important'
        },
        {
            'text': "$39,600 is the favored short-term target, whether or not Bitcoin price action ultimately returns to downward momentum.",
            'expected': 'Medium'
        },
        {
            'text': "The central People's Bank of China (PBoC)'s digital yuan is set to expand into the world of wealth management – and has also broken new ground in the field of education.",
            'expected': 'Important'
        },
        {
            'text': "A new report conducted by Tecnalia Research and Chainlink Labs asserts that blockchain and oracles can help fix climate issues.",
            'expected': 'Medium'
        }
    ]

    device = get_device()
    model = trainer.model.to(device)
    model.eval()

    print(" Test Results:")
    correct_predictions = 0

    for i, example in enumerate(test_examples, 1):
        if config.ENHANCE_PATTERNS:
            enhanced_text = enhance_text_with_patterns(example['text'])
        else:
            enhanced_text = example['text']

        clean_text = advanced_clean_text(enhanced_text)

        inputs = tokenizer(clean_text, truncation=True, padding=True,
                         max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(predictions[0]).item()
        confidence = predictions[0][pred_id].item()
        predicted = config.ID_TO_LABEL[pred_id]

        is_correct = predicted == example['expected']
        if is_correct:
            correct_predictions += 1

        status = "✅" if is_correct else "❌"
        print(f"\n{i}. {status}")
        print(f"   Text: {example['text'][:80]}...")
        print(f"   Expected: {example['expected']}")
        print(f"   Predicted: {predicted} (confidence: {confidence:.3f})")

        if config.ENHANCE_PATTERNS and enhanced_text != example['text']:
            print(f"   Enhanced: {enhanced_text[:80]}...")

    accuracy = correct_predictions / len(test_examples)
    print(f"\n New Examples Test Results:")
    print(f"   Accuracy: {correct_predictions}/{len(test_examples)} ({accuracy*100:.1f}%)")

def fine_tune_model(trainer, tokenizer, test_dataset):
    """Fine-tune the model for better performance."""
    if trainer is None:
        return None

    print(f"\n FINE-TUNING MODEL")
    print("=" * 50)

    # fine-tuning için veri setini yeniden yükle
    print(" Re-loading dataset for fine-tuning...")
    df = upload_and_load_1835_dataset()
    if df is None:
        return None

    # fine-tuning için gelişmiş desenler uygulayın
    def apply_enhanced_patterns(text, label):
        enhanced = text
        text_lower = text.lower()

        if re.search(r'\$\d+\.?\d*\s*billion', text_lower):
            enhanced = "[CRITICAL_IMP] " + enhanced
        elif re.search(r'\$\d+\.?\d*\s*million', text_lower):
            enhanced = "[MEDIUM_IMP] " + enhanced
        elif any(word in text_lower for word in ['SEC', 'government', 'regulatory']):
            enhanced = "[HIGH_IMP] " + enhanced
        elif any(word in text_lower for word in ['hack', 'exploit', 'breach']):
            enhanced = "[SECURITY] " + enhanced
        else:
            enhanced = "[LOW_PATTERN] " + enhanced

        return enhanced

    print(" Applying enhanced patterns for fine-tuning...")
    df['text'] = df.apply(lambda row: apply_enhanced_patterns(row['text'], row['label']), axis=1)

    # yeni dengeli veri kümeleri oluşturun
    train_dataset_ft, val_dataset_ft, test_dataset_ft, tokenizer_ft, _ = create_balanced_datasets(df)

    new_tokens = ["[CRITICAL_IMP]", "[HIGH_IMP]", "[MEDIUM_IMP]", "[SECURITY]", "[LOW_PATTERN]"]
    added_tokens = tokenizer_ft.add_tokens(new_tokens)
    print(f" Added {added_tokens} new tokens for fine-tuning")

    trainer.model.resize_token_embeddings(len(tokenizer_ft))

    # Fine-tuning argümanları
    fine_tune_args = TrainingArguments(
        output_dir="/content/fine_tuned_output",
        num_train_epochs=3,  # daha az epochs
        per_device_train_batch_size=max(1, config.BATCH_SIZE // 2),  # küçük batch
        per_device_eval_batch_size=max(1, config.BATCH_SIZE // 2),
        learning_rate=5e-6,  # düşük learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1",
        greater_is_better=True,

        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=25,
        save_total_limit=2,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],

        seed=42,
        data_seed=42,
    )

    fine_tune_trainer = WeightedTrainer(
        model=trainer.model,
        args=fine_tune_args,
        train_dataset=train_dataset_ft,
        eval_dataset=val_dataset_ft,
        tokenizer=tokenizer_ft,
        data_collator=DataCollatorWithPadding(tokenizer_ft, padding=True),
        compute_metrics=compute_advanced_metrics,
        class_weights=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f" Fine-tuning Configuration:")
    print(f"   Learning rate: {fine_tune_args.learning_rate}")
    print(f"   Epochs: {fine_tune_args.num_train_epochs}")
    print(f"   Batch size: {fine_tune_args.per_device_train_batch_size}")

    try:
        print(" Starting fine-tuning...")
        fine_tune_trainer.train()
        print(" Fine-tuning completed successfully!")

        print(" Evaluating fine-tuned model...")
        ft_results = fine_tune_trainer.evaluate(test_dataset_ft)

        print(f"\n FINE-TUNING RESULTS:")
        print(f"   Accuracy: {ft_results['eval_accuracy']:.4f}")
        print(f"   Macro F1: {ft_results['eval_macro_f1']:.4f}")
        print(f"   Weighted F1: {ft_results['eval_weighted_f1']:.4f}")

        return fine_tune_trainer, tokenizer_ft

    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return None, None

def save_model_for_production(trainer, tokenizer):
    """Save model with production-ready format for Colab."""
    if trainer is None:
        return

    print(f"\n SAVING PRODUCTION MODEL")
    print("=" * 50)

    production_path = "/content/crypto_model_production"
    os.makedirs(production_path, exist_ok=True)

    trainer.save_model(production_path)
    tokenizer.save_pretrained(production_path)

    config_dict = {
        'model_name': config.BASE_MODEL_NAME,
        'num_labels': len(config.LABEL_NAMES),
        'label_names': config.LABEL_NAMES,
        'label_to_id': config.LABEL_TO_ID,
        'id_to_label': config.ID_TO_LABEL,
        'enhance_patterns': config.ENHANCE_PATTERNS,
        'training_samples': 1835,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.NUM_EPOCHS
    }

    with open(f"{production_path}/training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    zip_path = '/content/crypto_model.zip'
    print(f" Model oluşturuluyor...")

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files_list in os.walk(production_path):
                for file in files_list:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, production_path)
                    zipf.write(file_path, f"crypto_model/{arcname}")

        files.download(zip_path)
        print(f"Model  indirildi!")

    except Exception as e:
        print(f"Model indirme hatası: {e}")

def main():
    """Main execution function for 1835 dataset training."""
    print("Optimized for Real-World Crypto News Classification")
    print("=" * 70)

    # Optimize edilmiş modeli eğitin
    trainer, tokenizer, test_dataset = train_optimized_model()

    if trainer is not None:
        detailed_evaluation(trainer, tokenizer, test_dataset)

        test_with_new_examples(trainer, tokenizer)

        fine_tune_choice = input(f"\n Do you want to fine-tune the model? (y/n): ")
        if fine_tune_choice.lower() == 'y':
            fine_tuned_trainer, fine_tuned_tokenizer = fine_tune_model(trainer, tokenizer, test_dataset)

            if fine_tuned_trainer is not None:
                print("\nTesting fine-tuned model with new examples...")
                test_with_new_examples(fine_tuned_trainer, fine_tuned_tokenizer)

                trainer = fine_tuned_trainer
                tokenizer = fine_tuned_tokenizer

        save_choice = input(f"\nSave model for production use? (y/n): ")
        if save_choice.lower() == 'y':
            save_model_for_production(trainer, tokenizer)

        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")

    else:
        print(f"Training failed!")

if __name__ == "__main__":
    main()