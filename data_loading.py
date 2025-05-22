import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

# Data Loading and Splitting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Reduce dataset size for faster simulation
DATASET_SIZE_USED = 2000
actual_dataset_size = min(DATASET_SIZE_USED, len(full_dataset))
if DATASET_SIZE_USED > len(full_dataset):
    print(f"Warning: Requested dataset size {DATASET_SIZE_USED} is greater than available {len(full_dataset)}. Using {len(full_dataset)} samples.")

reduced_dataset, _ = random_split(full_dataset, [actual_dataset_size, len(full_dataset) - actual_dataset_size],
                                  generator=torch.Generator().manual_seed(42)) # Seed for reproducibility

# Split labeled and unlabeled subsets
actual_num_labeled = min(NUM_LABELED, len(reduced_dataset))
if NUM_LABELED > len(reduced_dataset):
    print(f"Warning: NUM_LABELED {NUM_LABELED} exceeds reduced dataset size {len(reduced_dataset)}. "
          f"Using {len(reduced_dataset)} for labeled set, no unlabeled data will be left.")
    labeled_subset = reduced_dataset
    unlabeled_subset_data = Subset(reduced_dataset, []) # Empty subset
else:
    labeled_subset, unlabeled_subset_data = random_split(
        reduced_dataset,
        [actual_num_labeled, len(reduced_dataset) - actual_num_labeled],
        generator=torch.Generator().manual_seed(42) # Seed for reproducibility
    )
unlabeled_subset_size = len(unlabeled_subset_data)

clients = []
# Client 0 is the labeled client
clients.append(Client(client_id=0, dataset=labeled_subset, is_labeled=True))

# Distribute unlabeled data among the remaining clients
num_unlabeled_clients_to_create = NUM_CLIENTS - 1
if num_unlabeled_clients_to_create > 0 and unlabeled_subset_size > 0:
    base_split_size = unlabeled_subset_size // num_unlabeled_clients_to_create
    remainder = unlabeled_subset_size % num_unlabeled_clients_to_create
    lengths = [base_split_size + 1 if i < remainder else base_split_size for i in range(num_unlabeled_clients_to_create)]
    lengths = [l for l in lengths if l > 0] # Filter out zero lengths

    if sum(lengths) > 0 and sum(lengths) <= unlabeled_subset_size : # Ensure lengths match available data
        actual_unlabeled_subsets = random_split(unlabeled_subset_data, lengths, generator=torch.Generator().manual_seed(43))
        for i, subset in enumerate(actual_unlabeled_subsets):
            if len(clients) < NUM_CLIENTS:
                 clients.append(Client(client_id=len(clients), dataset=subset, is_labeled=False))
    else:
        print("Warning: Not enough unlabeled data to distribute among remaining clients as requested, or issue with split lengths.")
elif num_unlabeled_clients_to_create > 0 and unlabeled_subset_size == 0:
    print("Warning: No unlabeled data left to distribute among other clients.")

# Fill remaining client slots with empty datasets if NUM_CLIENTS is not met
while len(clients) < NUM_CLIENTS:
    print(f"Warning: Creating client {len(clients)} with empty dataset as not enough data was available for splitting.")
    clients.append(Client(client_id=len(clients), dataset=Subset(full_dataset, []), is_labeled=False))

print("Client Setup:")
for c in clients:
    loader_batches = 0
    if len(c.dataset) > 0:
        loader_batches = len(c.get_data_loader()) # Call get_data_loader to see batch count
    print(f"Client {c.client_id} | Labeled={c.has_labels()} | Samples={len(c)} | Batches (drop_last=True)={loader_batches}")
print("-" * 30)
