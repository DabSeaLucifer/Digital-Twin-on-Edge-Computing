# Server Initialization
generator = Generator(z_dim=Z_DIM).to(device)
discriminator = SemiSupervisedDiscriminator(
    fc_out_features_before_mbd=256,
    feature_drop_prob=FEATURE_DROP_PROB,
    mbd_out_features=MBD_OUT_FEATURES,
    mbd_kernel_dims=50
).to(device)

global_generator_state = generator.state_dict()
global_discriminator_state = discriminator.state_dict()

# For loss-based client selection & graphing
client_loss_history = {client_obj.client_id: float('inf') for client_obj in clients}
all_server_avg_D_losses = []
all_server_avg_G_losses = []

print(f"\nStarting Federated Training with FedProx (μ={MU}), FeatureDrop ({FEATURE_DROP_PROB}), MBD (out_B={MBD_OUT_FEATURES}, kernel_C=50)...")

# Federated Learning Loop
for round_idx in range(NUM_ROUNDS):
    print(f"\n=== Federated Round {round_idx+1}/{NUM_ROUNDS} ===")

    client_G_states_for_agg = []
    client_D_states_for_agg = []
    client_sample_counts_for_agg = []
    round_metrics_all_selected_clients = [] # Stores metrics for all *selected* clients this round

    # Client Selection
    selected_clients_this_round = []
    clients_with_data = [c for c in clients if len(c.dataset) > 0 and len(c.get_data_loader()) > 0]

    if not clients_with_data:
        print("  No clients with data (or enough data for a full batch with drop_last=True) to train. Ending training.")
        break

    num_selectable_clients = len(clients_with_data)

    if CLIENTS_PER_ROUND_ACTUAL >= num_selectable_clients or CLIENTS_PER_ROUND_ACTUAL <= 0:
        selected_clients_this_round = clients_with_data
        print(f"  Selected all {len(selected_clients_this_round)} clients with data.")
    else:
        print(f"  Selecting {CLIENTS_PER_ROUND_ACTUAL} clients using loss-based strategy...")
        # Separate labeled (client 0) and unlabeled clients that have data
        labeled_client = next((c for c in clients_with_data if c.client_id == 0), None)
        unlabeled_clients_with_data = [c for c in clients_with_data if c.client_id != 0]

        # trying to include the labeled client if it has data and we are selecting
        if labeled_client and len(selected_clients_this_round) < CLIENTS_PER_ROUND_ACTUAL:
            selected_clients_this_round.append(labeled_client)
            print(f"    Prioritized labeled client: {labeled_client.client_id}")

        remaining_slots = CLIENTS_PER_ROUND_ACTUAL - len(selected_clients_this_round)

        # Selecting from unlabeled clients if slots remain
        if remaining_slots > 0 and unlabeled_clients_with_data:
            # Filter out clients already selected (e.g. if labeled client was added)
            unlabeled_candidates = [c for c in unlabeled_clients_with_data if c not in selected_clients_this_round]

            if unlabeled_candidates:
                default_high_loss = float('inf')
                # Clients with no/NaN history get default_high_loss, effectively random among them if all are like that.
                sorted_unlabeled_candidates = sorted(
                    unlabeled_candidates,
                    key=lambda c: client_loss_history.get(c.client_id, default_high_loss) if not np.isnan(client_loss_history.get(c.client_id, default_high_loss)) else default_high_loss,
                    reverse=True
                )

                # If all candidates have default_high_loss (e.g., first round of selection), shuffle for randomness
                is_first_selection_round_or_no_history = all(
                    (client_loss_history.get(c.client_id, default_high_loss) == default_high_loss or
                     np.isnan(client_loss_history.get(c.client_id, default_high_loss)))
                    for c in sorted_unlabeled_candidates
                )
                if is_first_selection_round_or_no_history and len(sorted_unlabeled_candidates) > 1:
                    np.random.shuffle(sorted_unlabeled_candidates)
                    print("    Selecting unlabeled clients randomly (no valid/differentiating loss history).")

                num_to_select_from_unlabeled = min(remaining_slots, len(sorted_unlabeled_candidates))
                selected_unlabeled_from_sorted = sorted_unlabeled_candidates[:num_to_select_from_unlabeled]
                selected_clients_this_round.extend(selected_unlabeled_from_sorted)
                print(f"    Selected {len(selected_unlabeled_from_sorted)} unlabeled clients: {[c.client_id for c in selected_unlabeled_from_sorted]}")

        # If still haven't filled CLIENTS_PER_ROUND_ACTUAL (e.g., few unlabeled, or CLIENTS_PER_ROUND_ACTUAL is high)
        # filling randomly from any remaining clients_with_data not yet selected
        if len(selected_clients_this_round) < CLIENTS_PER_ROUND_ACTUAL:
            remaining_to_choose_from_any = [c for c in clients_with_data if c not in selected_clients_this_round]
            if remaining_to_choose_from_any:
                num_to_randomly_add = min(CLIENTS_PER_ROUND_ACTUAL - len(selected_clients_this_round), len(remaining_to_choose_from_any))
                randomly_added = np.random.choice(remaining_to_choose_from_any, size=num_to_randomly_add, replace=False).tolist()
                selected_clients_this_round.extend(randomly_added)
                print(f"    Additionally selected {len(randomly_added)} clients randomly to fill slots: {[c.client_id for c in randomly_added]}")

        print(f"  Final selected clients for this round ({len(selected_clients_this_round)}): {[c.client_id for c in selected_clients_this_round]}")


    if not selected_clients_this_round:
        print("  No clients selected for training this round (after selection logic). Ending training.")
        break

    current_round_global_gen_state = copy.deepcopy(global_generator_state)
    current_round_global_disc_state = copy.deepcopy(global_discriminator_state)

    for c_train in selected_clients_this_round:
        print(f"  Training Client {c_train.client_id} (Labeled: {c_train.has_labels()}, Samples: {len(c_train.dataset)})...")

        updated_local_gen_state, updated_local_disc_state, metrics = train_on_client_data_fedprox(
            client=c_train,
            initial_global_gen_state_dict=current_round_global_gen_state,
            initial_global_disc_state_dict=current_round_global_disc_state,
            mu=MU, local_epochs=LOCAL_EPOCHS,
            local_lr_g=LEARNING_RATE_G, local_lr_d=LEARNING_RATE_D,
            lambda_fm=LAMBDA_FM
        )
        round_metrics_all_selected_clients.append(metrics) # Store metrics for every selected client

        # Update loss history for selection in the *next* round
        if metrics["samples_processed"] > 0 and not (np.isnan(metrics['avg_loss_D']) or np.isinf(metrics['avg_loss_D'])):
            client_loss_history[metrics['client_id']] = metrics['avg_loss_D']
        elif metrics["samples_processed"] == 0 :
             client_loss_history[metrics['client_id']] = float('inf')


        # For aggregation, consider only successfully trained clients with valid losses for G and D
        if metrics["samples_processed"] > 0 and \
           not (np.isnan(metrics['avg_loss_G']) or np.isinf(metrics['avg_loss_G'])) and \
           not (np.isnan(metrics['avg_loss_D']) or np.isinf(metrics['avg_loss_D'])):
            print(f"    Client {metrics['client_id']} finished training. Avg Base LossD={metrics['avg_loss_D']:.4f}, Avg Base LossG={metrics['avg_loss_G']:.4f}, Samples={metrics['samples_processed']}, Epochs={metrics['epochs_performed']}")
            client_G_states_for_agg.append(updated_local_gen_state)
            client_D_states_for_agg.append(updated_local_disc_state)
            client_sample_counts_for_agg.append(metrics['samples_processed'])
        else:
            print(f"    Client {metrics['client_id']} had no data, resulted in invalid loss, or did not complete training meaningfully. Skipping its update for aggregation.")

    # Server Aggregation Phase (FedAvg/FedProx style)
    print("  Aggregating client updates...")
    total_samples_for_agg = sum(client_sample_counts_for_agg)

    if total_samples_for_agg > 0 and client_G_states_for_agg:
        avg_G_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_generator_state.items()}
        avg_D_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_discriminator_state.items()}

        for i in range(len(client_G_states_for_agg)): # Iterate through the collected valid updates
            weight = client_sample_counts_for_agg[i] / total_samples_for_agg

            # Aggregate Generator
            for k_g in avg_G_state.keys():
                avg_G_state[k_g] += (client_G_states_for_agg[i][k_g] * weight).to(avg_G_state[k_g].dtype)

            # Aggregate Discriminator
            for k_d in avg_D_state.keys():
                avg_D_state[k_d] += (client_D_states_for_agg[i][k_d] * weight).to(avg_D_state[k_d].dtype)

        global_generator_state = avg_G_state
        global_discriminator_state = avg_D_state
        print(f"  Aggregation complete based on {len(client_G_states_for_agg)} valid clients.")
    else:
        print("  No valid client updates this round or no samples processed by valid clients. Global model unchanged.")

    generator.load_state_dict(global_generator_state)
    discriminator.load_state_dict(global_discriminator_state)

    #  Evaluation / Sample Generation for this Round

    valid_metrics_d_this_round = [m['avg_loss_D'] for m in round_metrics_all_selected_clients if m['samples_processed'] > 0 and not np.isnan(m['avg_loss_D'])]
    valid_metrics_g_this_round = [m['avg_loss_G'] for m in round_metrics_all_selected_clients if m['samples_processed'] > 0 and not np.isnan(m['avg_loss_G'])]

    avg_round_loss_D_server = np.mean(valid_metrics_d_this_round) if valid_metrics_d_this_round else float('nan')
    avg_round_loss_G_server = np.mean(valid_metrics_g_this_round) if valid_metrics_g_this_round else float('nan')

    all_server_avg_D_losses.append(avg_round_loss_D_server)
    all_server_avg_G_losses.append(avg_round_loss_G_server)
    print(f"  Round {round_idx+1} Average Base Losses (among successfully trained selected clients): D={avg_round_loss_D_server:.4f}, G={avg_round_loss_G_server:.4f}")


    generator.eval()
    if Z_DIM > 0:
        fixed_noise_eval = torch.randn(8, Z_DIM, device=device)
        with torch.no_grad():
            samples = generator(fixed_noise_eval).cpu()
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        fig, axes = plt.subplots(1, 8, figsize=(15,2))
        for i in range(8):
            img_display = samples[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img_display)
            axes[i].axis('off')
        plt.suptitle(f"Generated Samples after Round {round_idx+1}")
        plt.show()
    else:
        print("  Z_DIM is 0 or negative, skipping sample generation.")
    generator.train()

#  Plotting server-side average losses after all rounds
print("\nPlotting server-side average losses per round...")
num_actual_rounds = len(all_server_avg_D_losses)
rounds_for_plot = range(1, num_actual_rounds + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

valid_d_losses_plot = [(r, l) for r, l in zip(rounds_for_plot, all_server_avg_D_losses) if not np.isnan(l)]
if valid_d_losses_plot:
    r_d, l_d = zip(*valid_d_losses_plot)
    plt.plot(r_d, l_d, marker='o', linestyle='-', label='Avg Discriminator Loss (Base)')
else:
    plt.text(0.5, 0.5, 'No valid D loss data to plot', horizontalalignment='center', verticalalignment='center')
plt.title('Server: Average Discriminator Loss per Round')
plt.xlabel('Federated Round')
plt.ylabel('Average Base Loss D')
if num_actual_rounds > 0: plt.xticks(rounds_for_plot)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
valid_g_losses_plot = [(r, l) for r, l in zip(rounds_for_plot, all_server_avg_G_losses) if not np.isnan(l)]
if valid_g_losses_plot:
    r_g, l_g = zip(*valid_g_losses_plot)
    plt.plot(r_g, l_g, marker='x', linestyle='-', color='r', label='Avg Generator Loss (Base)')
else:
    plt.text(0.5, 0.5, 'No valid G loss data to plot', horizontalalignment='center', verticalalignment='center')

plt.title('Server: Average Generator Loss per Round')
plt.xlabel('Federated Round')
plt.ylabel('Average Base Loss G')
if num_actual_rounds > 0: plt.xticks(rounds_for_plot)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nFederated Training with FedProx, SGAN components, Loss-based Client Selection, and Graphing complete.")
