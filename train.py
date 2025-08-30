import argparse
from Src.Training.Trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA Solver Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Initialize the trainer
    trainer = Trainer(
        data_dir=args.data_dir,
        num_classes=36,
        grid_height=20,
        grid_width=80,
        device=args.device,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Train the model
    train_losses = trainer.train(
        epochs=args.epochs,
        resume_from=args.resume,
        save_every=args.save_every
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    # For simple usage without command line arguments
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/train"
    
    trainer = Trainer(
        data_dir=data_dir,
        num_classes=36,
        grid_height=20,
        grid_width=80,
        device='cuda',
        optimizer_type='adam',
        learning_rate=0.001,
        weight_decay=1e-4,
        save_dir='./checkpoints'
    )
    
    # Train the model
    trainer.train(epochs=10, save_every=2)