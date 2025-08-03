import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss():
    def __init__(self, temperature=0.07, contrast_mode='all'):
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def __call__(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = (features @ features.T) / self.temperature

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        identity_mask = torch.eye(batch_size, device=device)
        mask = mask * (1 - identity_mask)

        positives_per_sample = mask.sum(dim=1)

        logits_mask = torch.ones_like(similarity_matrix) - identity_mask

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        loss_per_sample = - (mask * log_prob).sum(dim=1) / positives_per_sample.clamp(min=1)

        valid_samples = positives_per_sample > 0
        if valid_samples.sum() > 0:
            loss = loss_per_sample[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss