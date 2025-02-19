import torch
from model import Generator
from utils import decrypt_data, aggregate_weights

class FederatedServer:
    def __init__(self, device):
        self.global_generator = Generator().to(device)

    def aggregate_updates(self, client_updates):
        decrypted_updates = [decrypt_data(client["generator_weights"]) for client in client_updates]
        aggregated_weights = aggregate_weights(decrypted_updates)

        self.global_generator.load_state_dict(aggregated_weights)

    def distribute_model(self, clients):
        for client in clients:
            client.generator.load_state_dict(self.global_generator.state_dict())
