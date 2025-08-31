# TrainerTest.py
from Src.Training.Trainer import Trainer, TrainConfig

if __name__ == "__main__":
    cfg = TrainConfig(
        data_dir="/home/utn/abap44us/Downloads/UTN-CV25-Captcha-Dataset/part2/train",
        val_data_dir="/home/utn/abap44us/Downloads/UTN-CV25-Captcha-Dataset/part2/val",  # <- set to a real val dir, or None to reuse train
        num_classes=36,
        grid_height=20,
        grid_width=80,
        img_width=640,
        img_height=160,
        device="cuda",
        batch_size=32,
        lr=1e-3,
        epochs=1,
        grad_clip=1.0,
        use_amp=True,
        ckpt_dir="./checkpoints/exp1",
        log_dir="./checkpoints/exp1",  # TensorBoard + CSV here
        log_every=100,                 # set 0 to disable per-iter logs
        resume=True,                   # resume from last.pth if exists
    )

    Trainer(cfg).fit()
