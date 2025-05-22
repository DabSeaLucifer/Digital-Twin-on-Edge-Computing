import torch
import torch.optim as optim
import copy

def train_on_client_data_fedprox(client, initial_global_gen_state_dict, initial_global_disc_state_dict,
                                 mu, local_epochs, local_lr_g, local_lr_d, lambda_fm):
    if len(client.dataset) == 0 or len(client.get_data_loader()) == 0:
        metrics = {
            "client_id": client.client_id, "is_labeled": client.has_labels(),
            "avg_loss_D": float('nan'), "avg_loss_G": float('nan'),
            "samples_processed": 0, "epochs_performed": 0,
            "client_specific_lr_g": local_lr_g, "client_specific_lr_d": local_lr_d,
            "client_batch_size": BATCH_SIZE
        }

        return copy.deepcopy(initial_global_gen_state_dict), copy.deepcopy(initial_global_disc_state_dict), metrics

    local_generator = Generator(z_dim=Z_DIM).to(device)
    local_discriminator = SemiSupervisedDiscriminator(
        fc_out_features_before_mbd=256,
        feature_drop_prob=FEATURE_DROP_PROB,
        mbd_out_features=MBD_OUT_FEATURES,
        mbd_kernel_dims=50
    ).to(device)

    local_generator.load_state_dict(copy.deepcopy(initial_global_gen_state_dict))
    local_discriminator.load_state_dict(copy.deepcopy(initial_global_disc_state_dict))

    # Store initial global parameters for FedProx proximal term
    initial_gen_params_for_prox = [p.clone().detach() for p in local_generator.parameters()]
    initial_disc_params_for_prox = [p.clone().detach() for p in local_discriminator.parameters()]

    local_optim_D = optim.Adam(local_discriminator.parameters(), lr=local_lr_d, betas=(0.5, 0.999))
    local_optim_G = optim.Adam(local_generator.parameters(), lr=local_lr_g, betas=(0.5, 0.999))

    loader = client.get_data_loader()
    is_labeled = client.has_labels()
    total_loss_D_base, total_loss_G_base, batch_count = 0.0, 0.0, 0
    samples_processed_client = 0

    local_generator.train()
    local_discriminator.train()

    for epoch in range(local_epochs):
        for real_images, labels_or_ignore in loader:
            if len(real_images) < BATCH_SIZE and loader.drop_last == False:
                 continue # Skip incomplete batches if drop_last was False for some reason

            batch_count += 1
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            if current_batch_size == 0: continue # Should not happen with drop_last=True
            samples_processed_client += current_batch_size

            # Discriminator Update
            local_optim_D.zero_grad()
            base_loss_D_batch = 0.0

            # Loss for real images
            logits_real = local_discriminator(real_images)
            base_loss_D_batch += unsupervised_loss_real(logits_real)
            if is_labeled:
                # Note: Truncated here in the original document
                pass
