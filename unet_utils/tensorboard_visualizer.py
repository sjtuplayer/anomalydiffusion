import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
# Writer will output to ./runs/ directory by default

class TensorboardVisualizer():

    def __init__(self,log_dir='./logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def visualize_image_batch(self,image_batch,n_iter,image_name='Image_batch'):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name,grid,n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        self.writer.add_scalar(loss_name, loss_val, n_iter)

