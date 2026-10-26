import os

from src.phas_id.train_module import PHASIDTrainModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
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

    arg_parser.add_argument("-mn", "--model_name", type=str, default="microsoft/deberta-v3-base", help="Pretrained encoder model name (HuggingFace hub)")
    arg_parser.add_argument("-tl", "--token_limit", type=int, default=128, help="Token limit per sentence")
    arg_parser.add_argument("-K", "--K", type=int, default=5, help="Number of support documents per episode")
    arg_parser.add_argument("-ckpt", "--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt)")
    arg_parser.add_argument("-lp", "--log_path", type=str, default="logs/phas_id_eval", help="Path to save the evaluation logs")
    arg_parser.add_argument("-bo", "--binary_output", action="store_true", help="Collapse predictions to binary (Human vs AI/Human-AI) using softmax on first two logits")

    console.rule(f"[bold {ConsoleColors.yellow}]Starting Semi-Personalized Detector Evaluation")
    console.log("Parsing arguments")

    args = arg_parser.parse_args()

    seed_everything(args.seed, workers=True)

    console.log("Loading model from checkpoint")
    model = PHASIDTrainModule.load_from_checkpoint(
        args.checkpoint_path,
        run_name=os.path.splitext(os.path.basename(args.checkpoint_path))[0],
    )
    model.binary_output = args.binary_output
    num_classes = model.hparams["num_classes"]

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

    console.rule("[bold]Evaluation setting")
    console.print(f"Random seed:            {args.seed}")
    console.print(f"Dataset:                {args.dataset}")
    console.print(f"Setting type:           {setting_type}")
    console.print(f"Num classes (from ckpt):{num_classes}")
    console.print(f"Model name:             {args.model_name}")
    console.print(f"Token limit:            {args.token_limit}")
    console.print(f"K (support docs):       {args.K}")
    console.print(f"Checkpoint path:        {args.checkpoint_path}")
    console.print(f"Log path:               {args.log_path}")
    console.print(f"Binary output:          {args.binary_output}")
    console.print(f"Recreate:               {args.recreate}")
    console.rule()

    os.makedirs(args.log_path, exist_ok=True)

    run_name = model.run_name

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

    csv_logger = CSVLogger(os.path.join(args.log_path, run_name))

    console.log("Initializing trainer")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        logger=[csv_logger],
    )

    console.rule("[bold]Begin evaluation")
    trainer.test(model, datamodule=datamodule)

    console.rule("[bold]Evaluation complete")
