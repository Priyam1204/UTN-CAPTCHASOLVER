import os
from datetime import datetime
from Src.Training.Trainer import Trainer

def continue_training_safely():
    """Continue training with separate subdirectory to preserve original checkpoints"""
    
    # Original checkpoint location
    original_checkpoint = "./checkpoints/best_model.pth"
    
    # Create new subdirectory for continued training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    continue_dir = f"./checkpoints/continue_training_{timestamp}"
    os.makedirs(continue_dir, exist_ok=True)
    
    print(f"🔄 Continuing training from: {original_checkpoint}")
    print(f"💾 New checkpoints will be saved to: {continue_dir}")
    
    # Check if original checkpoint exists
    if not os.path.exists(original_checkpoint):
        print(f"❌ Original checkpoint not found: {original_checkpoint}")
        print("Available checkpoints:")
        if os.path.exists("./checkpoints"):
            for file in os.listdir("./checkpoints"):
                if file.endswith('.pth'):
                    print(f"  - {file}")
        return
    
    # Data directory
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/train"
    
    # Initialize trainer with NEW save directory
    trainer = Trainer(
        data_dir=data_dir,
        num_classes=36,
        grid_height=10,
        grid_width=40,
        device='cuda',
        optimizer_type='adam',
        learning_rate=0.00003,    # Will be overridden by checkpoint
        weight_decay=1e-4,
        save_dir=continue_dir     # ✅ NEW: Save to subdirectory
    )
    
    # Load the original checkpoint
    print("\n📥 Loading original checkpoint...")
    last_epoch = trainer.load_checkpoint(original_checkpoint)
    
    print(f"\n📊 Training Status:")
    print(f"  - Original training: Epochs 1-{last_epoch}")
    print(f"  - Continuing from: Epoch {last_epoch + 1}")
    print(f"  - Current best loss: {trainer.best_loss:.4f}")
    print(f"  - Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
    
    # ✅ COPY ORIGINAL CHECKPOINT to new directory for reference
    import shutil
    reference_path = os.path.join(continue_dir, "original_best_model.pth")
    shutil.copy2(original_checkpoint, reference_path)
    print(f"📋 Original checkpoint copied to: {reference_path}")
    
    # Continue training for 20 more epochs
    additional_epochs = 20
    target_epochs = last_epoch + additional_epochs
    
    print(f"\n🚀 Starting continued training:")
    print(f"  - Target epochs: {target_epochs} (adding {additional_epochs} more)")
    print(f"  - Checkpoints saved to: {continue_dir}")
    print(f"  - Original checkpoints preserved in: ./checkpoints/")
    
    # Train with new save directory
    trainer.train(epochs=target_epochs, save_every=5)
    
    print(f"\n✅ Continued training completed!")
    print(f"📁 Results saved in: {continue_dir}")
    print(f"   - best_model.pth (new best)")
    print(f"   - latest_checkpoint.pth (final state)")
    print(f"   - original_best_model.pth (reference)")
    print(f"📁 Original checkpoints preserved in: ./checkpoints/")

def check_training_progress():
    """Show progress of continued training"""
    print("📊 Training Progress Overview:")
    print()
    
    # Check original training
    original_checkpoint = "./checkpoints/best_model.pth"
    if os.path.exists(original_checkpoint):
        import torch
        checkpoint = torch.load(original_checkpoint, map_location='cpu')
        print(f"🏁 Original Training (./checkpoints/):")
        print(f"   - Completed: Epoch {checkpoint['epoch']}")
        print(f"   - Best Loss: {checkpoint['best_loss']:.4f}")
        print()
    
    # Check continued training directories
    continue_dirs = []
    if os.path.exists("./checkpoints"):
        for item in os.listdir("./checkpoints"):
            if item.startswith("continue_training_") and os.path.isdir(f"./checkpoints/{item}"):
                continue_dirs.append(item)
    
    if continue_dirs:
        print(f"🔄 Continued Training Sessions:")
        for dir_name in sorted(continue_dirs):
            dir_path = f"./checkpoints/{dir_name}"
            best_model = os.path.join(dir_path, "best_model.pth")
            
            if os.path.exists(best_model):
                import torch
                checkpoint = torch.load(best_model, map_location='cpu')
                print(f"   📁 {dir_name}:")
                print(f"      - Completed: Epoch {checkpoint['epoch']}")
                print(f"      - Best Loss: {checkpoint['best_loss']:.4f}")
            else:
                print(f"   📁 {dir_name}: (in progress or incomplete)")
        print()
    
    print("💡 Tip: Each continued training session creates its own subdirectory")
    print("💡      Your original ./checkpoints/best_model.pth is always preserved")

if __name__ == "__main__":
    print("="*60)
    print("🚀 CAPTCHA Model - Continue Training")
    print("="*60)
    
    # Show current status
    check_training_progress()
    
    print("\n" + "="*60)
    response = input("Continue with training? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        continue_training_safely()
    else:
        print("❌ Training cancelled.")