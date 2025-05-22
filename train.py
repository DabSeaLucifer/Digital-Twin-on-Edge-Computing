# FedProx Client Training Function
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
                labels = labels_or_ignore.to(device)
                base_loss_D_batch += supervised_loss(logits_real, labels, smoothing=SMOOTHING)

            # For Feature Matching (mean of features from real data before MBD and dropout)
            with torch.no_grad():
                real_feat_mean_detached = local_discriminator.extract_features(real_images).mean(dim=0).detach()

            # Loss for fake images
            noise = torch.randn(current_batch_size, Z_DIM, device=device)
            with torch.no_grad():
                fake_images = local_generator(noise).detach()
            logits_fake = local_discriminator(fake_images)
            base_loss_D_batch += unsupervised_loss_fake(logits_fake)

            # FedProx term for Discriminator
            prox_term_D = 0.0
            for local_param_d, initial_global_param_d in zip(local_discriminator.parameters(), initial_disc_params_for_prox):
                if local_param_d.requires_grad:
                     prox_term_D += torch.sum((local_param_d - initial_global_param_d.to(device))**2)
            loss_D_total = base_loss_D_batch + (MU / 2) * prox_term_D

            if torch.isnan(loss_D_total) or torch.isinf(loss_D_total):
                print(f"Warning: Invalid loss_D ({loss_D_total.item()}) on Client {client.client_id}. Skipping D update.")
            else:
                loss_D_total.backward()
                torch.nn.utils.clip_grad_norm_(local_discriminator.parameters(), max_norm=GRADIENT_CLIP_D)
                local_optim_D.step()
            total_loss_D_base += base_loss_D_batch.item()


            # Generator Update (potentially multiple times per D update)
            current_total_loss_G_batch_base = 0.0
            nan_in_g_update_step = False
            for _ in range(GENERATOR_UPDATES_PER_DISCRIMINATOR):
                 local_optim_G.zero_grad()
                 noise_g = torch.randn(current_batch_size, Z_DIM, device=device) # Use current_batch_size
                 fake_images_g = local_generator(noise_g)

                 logits_fake_for_G = local_discriminator(fake_images_g)
                 loss_G_adv = generator_loss_adversarial(logits_fake_for_G)

                 # Feature matching loss for G
                 fake_feat_mean_for_G = local_discriminator.extract_features(fake_images_g).mean(dim=0)
                 loss_G_fm_val = feature_matching_loss(real_feat_mean_detached, fake_feat_mean_for_G)
                 base_loss_G_batch_step = loss_G_adv + LAMBDA_FM * loss_G_fm_val

                 # FedProx term for Generator
                 prox_term_G = 0.0
                 for local_param_g, initial_global_param_g in zip(local_generator.parameters(), initial_gen_params_for_prox):
                      if local_param_g.requires_grad:
                          prox_term_G += torch.sum((local_param_g - initial_global_param_g.to(device))**2)
                 loss_G_total_step = base_loss_G_batch_step + (MU / 2) * prox_term_G

                 if torch.isnan(loss_G_total_step) or torch.isinf(loss_G_total_step):
                     print(f"Warning: Invalid loss_G ({loss_G_total_step.item()}) on Client {client.client_id}. Skipping this G update step.")
                     nan_in_g_update_step = True; break
                 else:
                    loss_G_total_step.backward()
                    torch.nn.utils.clip_grad_norm_(local_generator.parameters(), max_norm=GRADIENT_CLIP_G)
                    local_optim_G.step()
                    current_total_loss_G_batch_base += base_loss_G_batch_step.item()

            if not nan_in_g_update_step:
                total_loss_G_base += current_total_loss_G_batch_base / GENERATOR_UPDATES_PER_DISCRIMINATOR
            else:
                total_loss_G_base = float('nan')

    client.local_epochs_performed_last_round = local_epochs

    avg_loss_D_base_client = total_loss_D_base / batch_count if batch_count > 0 and not np.isnan(total_loss_D_base) else float('nan')
    avg_loss_G_base_client = total_loss_G_base / batch_count if batch_count > 0 and not np.isnan(total_loss_G_base) else float('nan')

    metrics = {
        "client_id": client.client_id,
        "is_labeled": is_labeled,
        "avg_loss_D": avg_loss_D_base_client,
        "avg_loss_G": avg_loss_G_base_client,
        "samples_processed": samples_processed_client,
        "epochs_performed": local_epochs,
        "client_specific_lr_g": local_lr_g,
        "client_specific_lr_d": local_lr_d,
        "client_batch_size": BATCH_SIZE
    }
    return local_generator.state_dict(), local_discriminator.state_dict(), metrics
