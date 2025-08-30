from Src.Training.Trainer import Trainer

if __name__ == "__main__":
    # Path to the dataset
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/train"
    
    # Initialize the trainer
    trainer = Trainer(data_dir=data_dir, num_classes=36, grid_height=20, grid_width=80, device='cuda')

    # Train the model for 1 epoch
    trainer.train(epochs=1)