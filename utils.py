from cryptography.fernet import Fernet
import torch
import copy
import io

key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_data(model_state):
    buffer = io.BytesIO()
    torch.save(model_state, buffer)
    encrypted_data = cipher.encrypt(buffer.getvalue())
    return encrypted_data

def decrypt_data(encrypted_data):
    decrypted_data = cipher.decrypt(encrypted_data)
    buffer = io.BytesIO(decrypted_data)
    return torch.load(buffer, weights_only=True)


def aggregate_weights(client_weights):
    global_model = copy.deepcopy(client_weights[0])
    for key in global_model.keys():
        global_model[key] = torch.mean(
            torch.stack([weights[key] for weights in client_weights]), dim=0
        )
    return global_model
