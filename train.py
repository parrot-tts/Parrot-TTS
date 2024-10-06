import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from modules import ParrotDataset, Parrot, ModelLoss
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import lightning as L
import yaml
import argparse
from pathlib import Path
import math

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LitParrot(L.LightningModule):
    
    # define model architecture
    def __init__(
        self, data_config, src_vocab_size, src_pad_idx
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = data_config
        self.parrot = Parrot(data_config, src_vocab_size, src_pad_idx)
        self.loss_fn = ModelLoss(data_config)

    # forward defines the prediction/inference actions
    def forward(self, batch, inference=False):
        return self.parrot(batch, inference=inference)

    # Training logic, calculate loss, and send logs to tensorboard and return loss
    def training_step(self, batch, batch_idx):
        out, _, _, log_dur_preds = self.parrot(batch)
        total_loss, code_loss, dur_loss = self.loss_fn(out, log_dur_preds, batch)

        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("train_code_loss", code_loss, prog_bar=True)
        self.log("train_dur_loss", dur_loss, prog_bar=True)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, rank_zero_only=True)

        # Clear GPU cache after every training step (if needed)
        torch.cuda.empty_cache()
        
        return total_loss

    # Validation logic, calculate loss, and send logs to tensorboard and return loss
    def validation_step(self, batch, batch_idx):
        out, _, _, log_dur_preds = self.parrot(batch)
        total_loss, code_loss, dur_loss = self.loss_fn(out, log_dur_preds, batch)

        self.log("val_total_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_code_loss", code_loss, prog_bar=True, sync_dist=True)
        self.log("val_dur_loss", dur_loss, prog_bar=True, sync_dist=True)
        return total_loss

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parrot.parameters(), 
            lr=self.train_config["optimizer"]["init_lr"],
            weight_decay=self.train_config["optimizer"]["weight_decay"]
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config["train"]["warmup_steps"],
            num_training_steps=self.train_config["train"]["total_steps"]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def infer(self, batch):
        self.eval()
        res = self.parrot.infer(batch)
        return res


def main(args):
    data_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # setup datasets
    train_dataset = ParrotDataset("train", data_config=data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["train"]["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
    )

    val_dataset = ParrotDataset("val", data_config=data_config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["train"]["batch_size"],
        collate_fn=val_dataset.collate_fn,
        num_workers=4,
    )
    src_vocab_size = train_dataset.src_vocab_size
    src_pad_idx = train_dataset.src_pad_idx

    # Init the lightning module
    lit_parrot = LitParrot(data_config ,src_vocab_size, src_pad_idx)

    # set up some model callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val_total_loss",
        filename="parrot_model-{step}-{val_total_loss_step:.2f}",
        mode="min",
        dirpath=data_config["path"]["root_path"]+"/ckpt",
        every_n_train_steps=data_config["train"]["save_every"],
    )

    log_path = Path(data_config["path"]["root_path"]) / "logs"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path / "tensorboard_logs")
    csv_logger = pl_loggers.CSVLogger(save_dir=log_path / "csv_logs")

    # automates all the hardware engineering
    trainer = L.Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=args.num_gpus,
        callbacks=[checkpoint_callback],
        max_steps=data_config["train"]["total_steps"],
        val_check_interval=data_config["train"]["val_every"],
        check_val_every_n_epoch=None,
        log_every_n_steps=data_config["train"]["log_every"],
        accumulate_grad_batches=data_config["train"]["grad_acc_steps"],
        gradient_clip_val=data_config["train"]["grad_clip"],
        deterministic=True,
        default_root_dir=data_config["path"]["log_path"],
        logger=[tb_logger, csv_logger],
    )

    # Clear GPU cache before starting training
    torch.cuda.empty_cache()

    # add pytorch dataloader
    trainer.fit(
        model=lit_parrot, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint_pth", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=2)

    args = parser.parse_args()
    L.seed_everything(42, workers=True)

    main(args)