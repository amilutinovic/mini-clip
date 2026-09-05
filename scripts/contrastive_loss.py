import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    image_embeddings,
    text_embeddings,
    temperature=0.07
):
    logits = (
        image_embeddings @ text_embeddings.T
    ) / temperature

    labels = torch.arange(
        logits.size(0),
        device=logits.device
    )

    image_to_text_loss = F.cross_entropy(
        logits,
        labels
    )

    text_to_image_loss = F.cross_entropy(
        logits.T,
        labels
    )

    loss = (
        image_to_text_loss +
        text_to_image_loss
    ) / 2

    return loss
