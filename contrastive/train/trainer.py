from pathlib import Path
import torch

from data.dataset import build_data
from model.Model import Model
from tqdm import tqdm

from train.logger_train import Logger

class Trainer:

    def __init__(self, task_name, data_params, train_params, model_params):
        self.model = Model(**model_params)
        data_params["processor"] = self.model.processor
        self.train_loader, self.val_loader = build_data(**data_params)
        self.build_optimizer(**train_params)
        self.logger = Logger()
        self.logger.add_metrics("Loss")

        self.log_path = Path(f"runs/{task_name}")
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.epochs = train_params["epochs"]

    def build_optimizer(self, lr, weight_decay, epochs, max_lr):
        self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(self.train_loader),
            epochs=epochs
        )

    def train_step(self):
        self.model.model.train()
        loss_train = 0.0
        for images, labels in tqdm(self.train_loader, total=len(self.train_loader), desc="Training Progress"):
            images, labels = images.cuda(), labels.cuda()
            images = images.view(-1, 3, images.shape[3], images.shape[4])
            labels = labels.view(-1)
            loss = self.model.loss(images, labels)
            loss_train += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss_train / len(self.train_loader)
    
    def val_step(self):
        self.model.model.train()
        loss_train = 0.0
        for images, labels in tqdm(self.val_loader, total=len(self.val_loader), desc="Training Progress"):
            images, labels = images.cuda(), labels.cuda()
            images = images.view(-1, 3, images.shape[3], images.shape[4])
            labels = labels.view(-1)
            loss = self.model.loss(images, labels)
            loss_train += loss.item()
        return loss_train / len(self.train_loader)

    def save_best(self, epoch):
        path = Path(f"{self.log_path}/{epoch}")
        path.mkdir(exist_ok=True, parents=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path.joinpath(f'best_model.pth'))

    def __call__(self):
        loss_train_best = 0.0
        for epoch in range(self.epochs):
            loss_train = self.train_step()
            loss_val = self.val_step()
            print(f"Loss train epoch {epoch}: {loss_train}")
            print(f"Loss val epoch {epoch}: {loss_val}")
            self.logger(epoch, "Loss", loss_val, loss_train)
            self.logger.save("Loss", self.log_path/"Loss.png")

            if loss_train_best < loss_train:
                self.save_best(epoch)


