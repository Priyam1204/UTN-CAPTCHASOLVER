import torch
import os
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR
from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer
from Src.Utils.CSVWriter import CSVLogger

class Trainer:
    def __init__(self, DataDir, NumClasses=36, GridHeight=10, GridWidth=40, 
                 Device='cuda', OptimizerType='adam', LearningRate=0.00003, 
                 WeightDecay=1e-4, SaveDir='./Checkpoints'):
        
        self.Device = torch.device(Device if torch.cuda.is_available() else 'cpu')
        self.SaveDir = SaveDir
        self.NumClasses = NumClasses
        self.GridHeight = GridHeight
        self.GridWidth = GridWidth
        self.CSV = CSVLogger(SaveDir)
        self.PrintedLossKeys = False
        os.makedirs(SaveDir, exist_ok=True)
        
        self.Model = CaptchaSolverModel(
            NumClasses=NumClasses, 
            GridHeight=GridHeight, 
            GridWidth=GridWidth
        ).to(self.Device)
        
        self.Criterion = ModelLoss(
            NumClasses=NumClasses, 
            GridHeight=GridHeight, 
            GridWidth=GridWidth
        ).to(self.Device)
        
        self.Optimizer = self.GetOptimizer(OptimizerType, LearningRate, WeightDecay)
        self.Scheduler = self.GetScheduler()
        self.TrainLoader = CaptchaDataLoader(DataDir, batch_size=8, shuffle=True)
        
        self.TargetPreparer = TargetPreparer(
            GridHeight=GridHeight, 
            GridWidth=GridWidth, 
            NumClasses=NumClasses,
            ImageWidth=640, 
            ImageHeight=160
        )
        
        self.BestLoss = float('inf')
        self.TrainLosses = []
        
    def GetOptimizer(self, OptimizerType, LearningRate, WeightDecay):
        """
        Initialize the optimizer based on the specified type.
        """
        if OptimizerType.lower() == 'adam':
            return Adam(self.Model.parameters(), lr=LearningRate, weight_decay=WeightDecay)
        elif OptimizerType.lower() == 'adamw':
            return AdamW(self.Model.parameters(), lr=LearningRate, weight_decay=WeightDecay)
        elif OptimizerType.lower() == 'sgd':
            return SGD(self.Model.parameters(), lr=LearningRate, momentum=0.9, weight_decay=WeightDecay)
        else:
            raise ValueError(f"Unsupported optimizer type: {OptimizerType}")
    
    def GetScheduler(self):
        """
        Initialize the learning rate scheduler.
        """
        return StepLR(self.Optimizer, step_size=3, gamma=0.8)
        
    def SaveCheckpoint(self, Epoch, Loss, IsBest=False):
        """
        Save model checkpoint.
        """
        Checkpoint = {
            'epoch': Epoch,
            'model_state_dict': self.Model.state_dict(),
            'optimizer_state_dict': self.Optimizer.state_dict(),
            'scheduler_state_dict': self.Scheduler.state_dict(),
            'loss': Loss,
            'best_loss': self.BestLoss,
            'train_losses': self.TrainLosses,
            'model_config': {
                'num_classes': self.NumClasses,
                'grid_height': self.GridHeight,
                'grid_width': self.GridWidth
            }
        }
        
        LatestPath = os.path.join(self.SaveDir, 'latest_checkpoint.pth')
        torch.save(Checkpoint, LatestPath)
        print(f"Saved checkpoint to {LatestPath}")
        
        if IsBest:
            BestPath = os.path.join(self.SaveDir, 'best_model.pth')
            torch.save(Checkpoint, BestPath)
            print(f"Saved best model to {BestPath}")
        
        EpochPath = os.path.join(self.SaveDir, f'checkpoint_epoch_{Epoch}.pth')
        torch.save(Checkpoint, EpochPath)
    
    def LoadCheckpoint(self, CheckpointPath):
        """
        Load model checkpoint.
        """
        if not os.path.exists(CheckpointPath):
            print(f"Checkpoint not found: {CheckpointPath}")
            return 0
        
        Checkpoint = torch.load(CheckpointPath, map_location=self.Device)
        
        self.Model.load_state_dict(Checkpoint['model_state_dict'])
        self.Optimizer.load_state_dict(Checkpoint['optimizer_state_dict'])
        self.Scheduler.load_state_dict(Checkpoint['scheduler_state_dict'])
        self.BestLoss = Checkpoint['best_loss']
        self.TrainLosses = Checkpoint['train_losses']
        
        Epoch = Checkpoint['epoch']
        Loss = Checkpoint['loss']
        
        print(f"Loaded checkpoint from epoch {Epoch} with loss {Loss:.4f}")
        return Epoch

    def TrainEpoch(self):
        """
        Train the model for one epoch.
        """
        self.Model.train()
        TotalLoss = 0
        from collections import defaultdict
        CompSums = defaultdict(float)
        OrderedKeys = None
        Preferred = ["total_loss", "bbox_loss", "obj_loss", "class_loss"]
        RunningTotal = 0.0
        NumBatches = len(self.TrainLoader)
        
        for BatchIdx, Batch in enumerate(self.TrainLoader):
            Images = Batch['Image'].to(self.Device)
            Targets = self.TargetPreparer(Batch).to(self.Device)
            
            Predictions = self.Model(Images)
        
            LossDict = self.Criterion(Predictions, Targets)
            Loss = LossDict['total_loss']
            ThisKeys = [K for K in LossDict.keys() if (K == "total_loss" or K.endswith("_loss"))]
            if OrderedKeys is None:
                OrderedKeys = [K for K in Preferred if K in ThisKeys] + [K for K in ThisKeys if K not in Preferred]
                self.CSV.init_batch_header(OrderedKeys)
            
            TotalLoss += Loss.item()
            
            self.Optimizer.zero_grad()
            Loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.Model.parameters(), max_norm=1.0)
            
            self.Optimizer.step()
            
            LossF = float(Loss.detach().item())
            RunningTotal += LossF
            RunningAvg = RunningTotal / (BatchIdx + 1)
            LrF = float(self.Optimizer.param_groups[0]['lr'])
            
            for K in OrderedKeys:
                V = LossDict[K]
                CompSums[K] += (float(V.detach().item()) if torch.is_tensor(V) else float(V))
            
            self.CSV.log_batch(
                epoch=getattr(self, "CurrentEpoch", -1),
                batch_idx=BatchIdx,
                num_batches=NumBatches,
                lr=LrF,
                loss_dict={K: (float(LossDict[K].detach().item()) if torch.is_tensor(LossDict[K]) else float(LossDict[K]))
                           for K in OrderedKeys},
                running_avg_total=RunningAvg
            )

            if BatchIdx % 20 == 0:
                TotalLossVal = Loss.item()
                ObjLossVal = LossDict.get('obj_loss', 0)
                BboxLossVal = LossDict.get('bbox_loss', 0)
                ClassLossVal = LossDict.get('class_loss', 0)
                
                print(f"Batch {BatchIdx}/{len(self.TrainLoader)}: "
                      f"Total={TotalLossVal:.4f}, "
                      f"Obj={ObjLossVal:.4f}, "
                      f"BBox={BboxLossVal:.4f}, "
                      f"Class={ClassLossVal:.4f}")
    
        AvgLoss = TotalLoss / len(self.TrainLoader)
        AvgTotal = CompSums['total_loss'] / max(1, NumBatches)
        AvgComponents = {K: (CompSums[K] / max(1, NumBatches)) for K in OrderedKeys if K != 'total_loss'}
        IsBest = AvgTotal < self.BestLoss
        if IsBest:
            self.BestLoss = AvgTotal
        
        self.CSV.init_epoch_header(OrderedKeys)
        self.CSV.log_epoch(
            epoch=getattr(self, "CurrentEpoch", -1),
            lr=float(self.Optimizer.param_groups[0]['lr']),
            is_best=IsBest,
            avg_total_loss=AvgTotal,
            avg_components=AvgComponents
        )

        print(f"\nEpoch Summary:")
        print(f"  Average Loss: {AvgLoss:.4f}")
        
        return AvgLoss
    
    def Train(self, Epochs, ResumeFrom=None, SaveEvery=5):
        """
        Train the model for multiple epochs.
        """
        StartEpoch = 0
        
        if ResumeFrom:
            StartEpoch = self.LoadCheckpoint(ResumeFrom)
        
        print(f"Starting training from epoch {StartEpoch + 1} to {Epochs}")
        print(f"Device: {self.Device}")
        print(f"Model parameters: {sum(P.numel() for P in self.Model.parameters()):,}")
        
        for Epoch in range(StartEpoch, Epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {Epoch + 1}/{Epochs}")
            print(f"{'='*50}")
            
            self.CurrentEpoch = Epoch + 1
            EpochLoss = self.TrainEpoch()
            
            self.Scheduler.step()
            
            self.TrainLosses.append(EpochLoss)
            
            IsBest = EpochLoss < self.BestLoss
            if IsBest:
                self.BestLoss = EpochLoss
            
            print(f"Epoch {Epoch + 1}/{Epochs} completed:")
            print(f"  Average Loss: {EpochLoss:.4f}")
            print(f"  Best Loss: {self.BestLoss:.4f}")
            print(f"  Learning Rate: {self.Optimizer.param_groups[0]['lr']:.6f}")
            
            if (Epoch + 1) % SaveEvery == 0 or IsBest or Epoch == Epochs - 1:
                self.SaveCheckpoint(Epoch + 1, EpochLoss, IsBest)
        
        print(f"\nTraining completed! Best loss: {self.BestLoss:.4f}")
        return self.TrainLosses
