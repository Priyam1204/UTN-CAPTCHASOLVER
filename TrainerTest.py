# TrainerTest.py (or your runner script)
from Src.Training.Trainer import Trainer

if __name__ == "__main__":
    data_dir = "/home/utn/abap44us/Downloads/UTN-CV25-Captcha-Dataset/part2/train"

    trainer = Trainer(
        data_dir=data_dir,
        num_classes=36,
        grid_height=20,
        grid_width=80,
        device="cuda",
        batch_size=32,
        lr=1e-3,
        ckpt_dir="./checkpoints",
    )

    trainer.train(epochs=20)
