import argparse
from Src.Training.Trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA Solver Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=2, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    trainer = Trainer(
        DataDir=args.data_dir,
        NumClasses=36,
        GridHeight=10,
        GridWidth=40,
        Device=args.device,
        OptimizerType=args.optimizer,
        LearningRate=args.lr,
        WeightDecay=args.weight_decay,
        SaveDir=args.save_dir
    )
    
    TrainLosses = trainer.Train(
        Epochs=args.epochs,
        ResumeFrom=args.resume,
        SaveEvery=args.save_every
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()