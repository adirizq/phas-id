import os

from src.phas_id.train_module import PHASIDTrainModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from uuid import uuid4

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.phas_id.data_module import PHASIDDataModule
from utils.console import ConsoleColors, console

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    arg_parser.add_argument("-ra_p", "--raw_data_path", type=str, default="dataset/coauthor/raw.csv", help="Path to the raw data")
    arg_parser.add_argument("-pa_p", "--parsed_data_path", type=str, default="dataset/coauthor/parsed_phas_id.jsonl", help="Path to the parsed data")
    arg_parser.add_argument("-d", "--dataset", type=str, default="coauthor", choices=["coauthor", "hacodet"], help="Dataset to use")
    arg_parser.add_argument("-r", "--recreate", action="store_true", help="Recreate dataset")
    arg_parser.add_argument("-p", "--patience", type=int, default=3, help="Patience for early stopping")

    arg_parser.add_argument("-mn", "--model_name", type=str, default="microsoft/deberta-v3-base", help="Pretrained encoder model name (HuggingFace hub)")
    arg_parser.add_argument("-uln", "--unfreeze_last_n", type=int, default=2, help="Number of encoder layers to unfreeze from the top")
    arg_parser.add_argument("-tl", "--token_limit", type=int, default=128, help="Token limit per sentence")
    arg_parser.add_argument("-me", "--max_epochs", type=int, default=20, help="Maximum number of epochs")
    arg_parser.add_argument("-lr", "--lr", type=float, default=2e-5, help="Learning rate")
    arg_parser.add_argument("-dr", "--dropout", type=float, default=0.0, help="Dropout rate")
    arg_parser.add_argument("-dal", "--doc_attn_layers", type=int, default=1, help="Number of intra-document attention layers")
    arg_parser.add_argument("-K", "--K", type=int, default=5, help="Number of support documents per episode")
    arg_parser.add_argument("-sp", "--model_save_path", type=str, default="weights/phas_id", help="Path to save the model")
    arg_parser.add_argument("-lp", "--log_path", type=str, default="logs/phas_id", help="Path to save the logs")

    console.rule(f"[bold {ConsoleColors.yellow}]Starting Semi-Personalized Detector Training")
    console.log("Parsing arguments")

    args = arg_parser.parse_args()

    seed_everything(args.seed, workers=True)

    if args.dataset == "coauthor":
        num_classes = 3
    elif args.dataset == "hacodet":
        num_classes = 2
    else:
        num_classes = 3

    if "unseen" in args.raw_data_path:
        setting_type = "Unseen Author"
    elif "global_author" in args.raw_data_path:
        setting_type = "Global Author"
    else:
        setting_type = "Seen Author"

    if "cross_domain_arg_to_cre" in args.raw_data_path:
        setting_type += " + Cross-Domain Argumentative-to-Creative"

    if "cross_domain_cre_to_arg" in args.raw_data_path:
        setting_type += " + Cross-Domain Creative-to-Argumentative"

    console.rule("[bold]Training setting")
    console.print(f"Random seed:            {args.seed}")
    console.print(f"Dataset:                {args.dataset}")
    console.print(f"Setting type:           {setting_type}")
    console.print(f"Model name:             {args.model_name}")
    console.print(f"Unfreeze last N:        {args.unfreeze_last_n}")
    console.print(f"Document attn layers:   {args.doc_attn_layers}")
    console.print(f"Early Stop Patience:    {args.patience}")
    console.print(f"Token limit:            {args.token_limit}")
    console.print(f"Max epochs:             {args.max_epochs}")
    console.print(f"Learning rate:          {args.lr}")
    console.print(f"Dropout:                {args.dropout}")
    console.print(f"K (support docs):       {args.K}")
    console.print(f"Model save path:        {args.model_save_path}")
    console.print(f"Log path:               {args.log_path}")
    console.print(f"Recreate:               {args.recreate}")
    console.rule()

    console.log("Setting up directories")
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    run_name = str(uuid4())[:8]

    console.log("Initializing datamodule")
    datamodule = PHASIDDataModule(
        raw_data_path=args.raw_data_path,
        parsed_data_path=args.parsed_data_path,
        model_name=args.model_name,
        token_limit=args.token_limit,
        K=args.K,
        num_classes=num_classes,
        recreate=args.recreate,
    )

    console.log("Initializing model")
    model = PHASIDTrainModule(
        model_name=args.model_name,
        num_classes=num_classes,
        K=args.K,
        unfreeze_last_n=args.unfreeze_last_n,
        num_intra_doc_layers=args.doc_attn_layers,
        lr=args.lr,
        run_name=run_name,
        dropout=args.dropout,
    )

    console.log("Initializing callbacks")
    early_stopping = EarlyStopping(
        monitor="val/macro_f1_score",
        patience=args.patience,
        mode="max",
        min_delta=0.001,
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.model_save_path, run_name),
        monitor="val/macro_f1_score",
        mode="max",
        save_top_k=1,
    )
    csv_logger = CSVLogger(os.path.join(args.log_path, run_name))

    console.log("Initializing trainer")
    trainer = Trainer(
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=8,
        callbacks=[early_stopping, checkpoint],
        logger=[csv_logger],
        log_every_n_steps=10,
    )

    console.rule("[bold]Begin training")
    trainer.fit(model, datamodule=datamodule)

    console.rule("[bold]Begin testing")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    console.rule("[bold]Finishing Experiment")
