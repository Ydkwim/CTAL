import os
import yaml
import argparse

from transformers import DataCollatorForLanguageModeling
from transformers import RobertaTokenizer, RobertaConfig
from transformers import RobertaForMaskedLM, LineByLineTextDataset

from transformers import Trainer, TrainingArguments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser for RoBERTa Models of the M2P project.')
    parser.add_argument('--config', default=None, type=str, help='Path to experiment config.')
    parser.add_argument('--name', default=None, type=str, help='Name for logging.')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    tokenizer = RobertaTokenizer.from_pretrained(
        '../tokenizer/libri-roberta_train-960'
    )
    
    model_config = RobertaConfig(
        vocab_size=30_000,
        max_position_embeddings=514,
        num_attention_heads=config['semantic']['num_attention_heads'],
        num_hidden_layers=config['semantic']['num_hidden_layers'],
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=model_config)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="/dataset/libri_language/V1_0/libri_language/train_960.txt",
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=os.path.join("result/result_language/",args.name),
        logging_dir=os.path.join("result/result_language/",args.name),
        overwrite_output_dir=True,
        num_train_epochs=config['runner']['num_epochs'],
        per_device_train_batch_size=config['dataloader']['batch_size'],
        save_steps=config['runner']['save_step'],
        logging_steps=config['runner']['log_step'],
        save_total_limit=config['runner']['max_keep'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(os.path.join("result/result_language/",args.name))