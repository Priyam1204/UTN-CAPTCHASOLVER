import torch

def CaptchaCollateFn(batch, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Collate function for CTC training.
    Converts CaptchaString into concatenated Targets + TargetLengths.
    """
    images = torch.stack([item['Image'] for item in batch])
    image_ids = [item['ImageID'] for item in batch]
    captcha_strings = [item['CaptchaString'] for item in batch]

    # Build char → index mapping
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    # Encode targets
    targets = []
    target_lengths = []
    for text in captcha_strings:
        encoded = [char_to_idx[ch] for ch in text if ch in char_to_idx]
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {
        "Image": images,                # (B, 1, H, W)
        "ImageID": image_ids,
        "CaptchaString": captcha_strings,
        "Targets": targets,             # concatenated 1D tensor
        "TargetLengths": target_lengths # (B,)
    }
