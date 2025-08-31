from torch.utils.data import DataLoader
from .DataSet import CaptchaDataset
from .Transform import Transform
from .Collate import CaptchaCollateFn

def CaptchaDataLoader(data_dir, batch_size=32, shuffle=True, num_workers=0, use_geo_aug=False):
    """
    Creates a DataLoader for the CAPTCHA dataset.

    Args:
        data_dir (str): Path to the dataset folder.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = CaptchaDataset(
        data_dir,
        transform=Transform(),
        use_geo_aug=use_geo_aug
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CaptchaCollateFn
    )