from server import FederatedServer
from client import FederatedClient
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

server = FederatedServer(device)
clients = [FederatedClient(i, device) for i in range(3)]  # no. of FL Clients

num_rounds = 5

for round_num in range(num_rounds):
    print(f"\n--- Federated Training Round {round_num+1} ---")

    client_updates = [client.train(epochs=2) for client in clients]
    server.aggregate_updates(client_updates)
    server.distribute_model(clients)
