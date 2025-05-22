import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=5): # kernel_dims is part of MBD_OUT_FEATURES
        super(MiniBatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.T = nn.Linear(in_features, out_features * kernel_dims, bias=False)

    def forward(self, x):
        # x is (batch_size, in_features)
        # M_i = T(h_i) -> (batch_size, out_features * kernel_dims)
        projected_x = self.T(x)
        # Reshape to (batch_size, out_features, kernel_dims)
        reshaped_x = projected_x.view(-1, self.out_features, self.kernel_dims)

        diffs = torch.abs(reshaped_x.unsqueeze(1) - reshaped_x.unsqueeze(0))
        l1_dist = diffs.sum(dim=3) # (batch_size, batch_size, out_features)

        mbd_feats_b = torch.exp(-l1_dist).sum(dim=1) - torch.exp(torch.zeros(1, device=x.device)) # (batch_size, out_features)

        return torch.cat([x, mbd_feats_b], dim=1)

class SemiSupervisedDiscriminator(nn.Module):
    def __init__(self, input_dim=3*32*32, num_classes=NUM_CLASSES,
                 fc_out_features_before_mbd=256,
                 feature_drop_prob=FEATURE_DROP_PROB,
                 mbd_out_features=MBD_OUT_FEATURES,
                 mbd_kernel_dims=50):
        super(SemiSupervisedDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.fc_out_features_before_mbd = fc_out_features_before_mbd

        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, fc_out_features_before_mbd),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Feature Drop Layer
        self.feature_drop = nn.Dropout(feature_drop_prob)

        self.mbd_layer = MiniBatchDiscrimination(fc_out_features_before_mbd, mbd_out_features, mbd_kernel_dims)

        # Input to final layer: fc_out_features_before_mbd (original) + mbd_out_features (from MBD)
        final_input_dim = fc_out_features_before_mbd + mbd_out_features
        self.final_classification_layer = nn.Linear(final_input_dim, num_classes + 1)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        features_extracted = self.feature_extractor_fc(x_flat)

        features_dropped = self.feature_drop(features_extracted)

        features_augmented_mbd = self.mbd_layer(features_dropped)

        logits = self.final_classification_layer(features_augmented_mbd)
        return logits

    def extract_features(self, x): # For feature matching (before feature drop & MBD)
        x_flat = x.view(x.size(0), -1)
        feats = self.feature_extractor_fc(x_flat)
        return feats

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, output_dim=3*32*32):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.fc(z)
        return img_flat.view(-1, 3, 32, 32)

# Loss Functions
def supervised_loss(logits, labels, smoothing=SMOOTHING):
    class_logits = logits[:, :NUM_CLASSES] # First NUM_CLASSES logits are for classification
    log_probs = F.log_softmax(class_logits, dim=1)
    true_dist = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)
    if smoothing > 0:
        true_dist = true_dist * (1 - smoothing) + smoothing / NUM_CLASSES
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# Real loss
def unsupervised_loss_real(logits): # D wants to output "not fake" for real images
    # Probability that the sample is fake (logit at FAKE_CLASS_IDX)
    # We want log(1 - P(fake|real_data)) to be maximized, or -log(1 - P(fake|real_data)) to be minimized
    prob_fake_for_real = F.softmax(logits, dim=1)[:, FAKE_CLASS_IDX]
    loss = -torch.log(1.0 - prob_fake_for_real + 1e-8).mean()
    return loss

# Fake loss
def unsupervised_loss_fake(logits): # D wants to output "fake" for fake images
    # We want log(P(fake|fake_data)) to be maximized, or -log(P(fake|fake_data)) to be minimized
    prob_fake_for_fake = F.softmax(logits, dim=1)[:, FAKE_CLASS_IDX]
    loss = -torch.log(prob_fake_for_fake + 1e-8).mean()
    return loss

def generator_loss_adversarial(logits_fake): # G wants D to output "not fake" for fake images
    # G wants to maximize log(1 - P(fake|fake_data_from_G))
    prob_fake_for_fake_from_G = F.softmax(logits_fake, dim=1)[:, FAKE_CLASS_IDX]
    loss = -torch.log(1.0 - prob_fake_for_fake_from_G + 1e-8).mean()
    return loss

def feature_matching_loss(real_features_mean, fake_features_mean):
    return F.mse_loss(fake_features_mean, real_features_mean)
