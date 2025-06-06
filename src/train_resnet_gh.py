import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import torchvision
from tqdm import tqdm
from typing import Type, Optional
from dataclasses import dataclass
from utils import get_resnet_for_fine_tuning, make_weigths_for_balanced_classes
import wandb
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve, precision_recall_curve, matthews_corrcoef
import argparse
import numpy as np
import random

device = 'cuda' if t.cuda.is_available() else 'cpu'

def compute_auroc(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def compute_auprc(y_true, y_score) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

@dataclass
class TrainingArgs:
    img_train_path: str = ""
    img_test_path: str = ""
    train_batch_size: int = 4
    test_batch_size: int = 64
    num_classes: int = 2
    epochs: int = 3
    lr: float = 3e-5
    model_name: str = ''
    outdir: str = ''
    wandb_project: Optional[str] = ''
    wandb_name: Optional[str] = ""

class ResNetTrainer:

    def __init__(self, args: TrainingArgs):
        self.args = args
        self.model = get_resnet_for_fine_tuning(self.args.num_classes).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr = self.args.lr)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_data = torchvision.datasets.ImageFolder(self.args.img_train_path, transform=self.transform)
        self.test_data = torchvision.datasets.ImageFolder(self.args.img_test_path, transform=self.transform)
        self.step = 0
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        wandb.watch(self.model.fc, log='all', log_freq=20)

    def train_dataloader(self):
        return data.DataLoader(self.train_data, batch_size=self.args.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.test_data, batch_size=self.args.test_batch_size, shuffle=False)

    def _shared_train_val_step(self, imgs, labels):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        return logits, labels
    
    def training_step(self, imgs, labels):
        logits, labels = self._shared_train_val_step(imgs, labels)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return loss
    
    @t.inference_mode
    def val_step(self, imgs, labels):
        logits, labels = self._shared_train_val_step(imgs, labels) 
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        loss = self.loss_fn(logits, labels)
        return predictions.cpu(), probabilities.cpu(), labels.cpu(), loss.cpu().item()

    
    def training_loop(self) -> int:
        
        self.model.train()
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        progress_bar = tqdm(total = self.args.epochs * len(self.train_data) // self.args.train_batch_size)
        accuracy = t.nan
        best_accuracy = 0    

        for epoch in range(self.args.epochs):
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                wandb.log({'loss': loss.item()}, step=self.step)
                progress_bar.set_description(f"Epoch: {epoch}, Loss: {loss}")
                progress_bar.update()

            # Validation step
            all_predictions, all_probabilities, all_labels = [], [], []
            val_loss = 0
            for imgs, labels in self.val_dataloader():
                preds, probs, lbls, v_loss = self.val_step(imgs, labels)
                all_predictions.append(preds)
                all_probabilities.append(probs)
                all_labels.append(lbls)
                val_loss += v_loss

            all_predictions = t.cat(all_predictions).numpy()
            all_probabilities = t.cat(all_probabilities).numpy()
            all_labels = t.cat(all_labels).numpy()

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
            auroc = compute_auroc(all_labels, all_probabilities)
            auprc = compute_auprc(all_labels, all_probabilities)
            accuracy = (all_labels == all_predictions).mean()
            mcc = matthews_corrcoef(all_labels, all_predictions)

            wandb.log({"accuracy": accuracy,
                       "precision": precision,
                       "recall": recall,
                       "f1_score": f1,
                       "auroc": auroc,
                       "auprc": auprc,
                       'mcc': mcc}, step = self.step)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                t.save(self.model.state_dict(), os.path.join(self.args.outdir, self.args.model_name))
                print(f"Model saved in {self.args.outdir} with accuracy {accuracy}")

            scheduler.step(1 - accuracy)

        wandb.finish()

def create_resnet_args(tf_name: str, batch_size: int) -> TrainingArgs:
    return TrainingArgs(
        img_train_path=f"dataset/images/{tf_name}/train",
        img_test_path=f"dataset/images/{tf_name}/test",
        outdir=f"models/ResNet/{tf_name}",
        wandb_project=f"ResNet-MCC",
        wandb_name=f"ResNet_{tf_name}_1bs_fullft",
        model_name=f"ResNet_{tf_name}_1bs_fullft.pth",
        train_batch_size=batch_size
    )

def main():
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, required=True, help="Name of the transcription factor (e.g. PATZ1)")
    parser.add_argument("--batch_size", type=int, required=True, help="train batch size")
    args = parser.parse_args()

    rn_args = create_resnet_args(args.tf_name, args.batch_size)
    os.makedirs(rn_args.outdir, exist_ok=True)

    trainer = ResNetTrainer(rn_args)
    trainer.training_loop()

if __name__ == '__main__':
    main()
