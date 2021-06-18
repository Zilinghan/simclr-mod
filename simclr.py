# Comments on `simclr`:
# Basically, it is the whole pipeline to train the SimCLR network.

import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F   # we only utilize the `F.normalize` in the code
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']                        # self.args contains all the user-input arguments
        self.model = kwargs['model'].to(self.args.device) # get the network model and move the model into GPU
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # It is the criterion for updating the parameters of nn, i.e., the loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device) 

    def classwise_contrastive_loss(self, features, labels):
        '''
        classwise_contrastive_loss:
                This function defines the classwise contrastive loss
        Inputs:
                features: It is the trained representations of the labeled batch
                labels: It is the labels for the labeled batch
        Outputs:
                loss: classwise contrastive loss
        '''
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.exp(torch.matmul(features, features.T)/self.args.temperature)
        similarity_total_sum = torch.sum(similarity_matrix, dim=1) - np.exp(1/self.args.temperature) 
        mask = (labels == labels.view(labels.shape[0], -1))
        similarity_class_sum = torch.sum((similarity_matrix * mask), dim=1) - np.exp(1/self.args.temperature)
        loss = -1* (torch.log(similarity_class_sum/similarity_total_sum)/torch.sum(mask, dim=1))
        return loss


    def info_nce_loss(self, features):
        ''' 
        info_nce_loss: 
                This function defines the instance-wise contrastive loss
        Inputs: 
                features: It is the trained representations of the unlabeled batch
        Outputs: 

        '''
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # labels looks like 
        # ---                               ---
        # | 1, 0, 0, ... ..., 1, 0, 0, ... ...|
        # | 0, 1, 0, ... ..., 0, 1, 0, ... ...|
        # | 0, 0, 1, ... ..., 0, 0, 1, ... ...|
        # | .                                .|
        # | .                                .|
        # | .                                .|
        # ---                               ---
        features = F.normalize(features, dim=1) # normalize the features along the row

        similarity_matrix = torch.matmul(features, features.T) # shape (2N, 2N)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device) # identity matrix
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Move all the predicted value to the first place (0-th position)
        logits = torch.cat([positives, negatives], dim=1)
        # Therefore, each label is 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, contrastive_train_loader, super_train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        for epoch_counter in range(self.args.epochs):
            for contra_images, _, super_images, super_labels in zip(tqdm(contrastive_train_loader), tqdm(super_train_loader)):
                contra_images = torch.cat(contra_images, dim=0)
                super_images = torch.cat(super_images, dim=0)

                contra_images = contra_images.to(self.args.device)
                super_images = super_images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    contra_features = self.model(contra_images)
                    super_features = self.model(super_images)
                    contra_logits, contra_labels = self.info_nce_loss(contra_features)
                    super_loss = self.classwise_contrastive_loss(super_features, super_labels)
                    loss = self.criterion(contra_logits, contra_labels) + super_loss

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(contra_logits, contra_labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
