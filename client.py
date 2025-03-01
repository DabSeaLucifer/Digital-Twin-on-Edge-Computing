import torch
import torch.optim as optim
import torch.nn as nn
from model import Generator, Discriminator
from dataset import get_cifar10_datasets, get_data_loader
from utils import encrypt_data

class FederatedClient:
    def __init__(self, client_id, device):
        self.client_id = client_id
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_dataset, _ = get_cifar10_datasets()
        self.train_loader = get_data_loader(self.train_dataset, batch_size=32)

    def train(self, epochs=5):
        """ Train SGAN locally and send model updates to the server. """
        total_loss_G, total_loss_D, batch_count = 0.0, 0.0, 0
        for epoch in range(epochs):
            for real_images, _ in self.train_loader:
                real_images = real_images.to(self.device)

                # Discriminator training
                self.optim_D.zero_grad()
                fake_images = self.generator(torch.randn(real_images.size(0), 100, device=self.device))
                real_preds = self.discriminator(real_images)
                fake_preds = self.discriminator(fake_images.detach())

                loss_D = -torch.mean(real_preds) + torch.mean(fake_preds)
                loss_D.backward()
                self.optim_D.step()

                # Generator training
                self.optim_G.zero_grad()
                fake_preds = self.discriminator(fake_images)
                loss_G = -torch.mean(fake_preds)
                loss_G.backward()
                self.optim_G.step()
                total_loss_D += loss_D.item()
                total_loss_G += loss_G.item()
                batch_count += 1

        avg_loss_G = total_loss_G / batch_count if batch_count > 0 else 0
        avg_loss_D = total_loss_D / batch_count if batch_count > 0 else 0

        return {
            "generator_weights": encrypt_data(self.generator.state_dict()),
            "discriminator_weights": encrypt_data(self.discriminator.state_dict()),
            "client_id": self.client_id,
            "avg_loss_G": avg_loss_G,
            "avg_loss_D": avg_loss_D,
            "samples_trained": len(self.train_dataset)
        }
