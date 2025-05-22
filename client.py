from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, dataset, is_labeled=False):
        self.client_id = client_id
        self.dataset = dataset
        self.is_labeled = is_labeled
        self.local_epochs_performed_last_round = 0 # For tracking

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=2 if device.type == 'cuda' else 0,
                          pin_memory=True if device.type == 'cuda' else False)

    def has_labels(self):
        return self.is_labeled

    def __len__(self):
        return len(self.dataset)
