
import torch
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device,scheduler = None,  early_stop=False, patience_limit=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.early_stop = early_stop
        self.patience_limit = patience_limit
        self.best_val_loss = float('inf') 
        self.best_model = None
        self.patience_check = 0  
        self.save_path = './best_model.pt'  
        self.scheduler = scheduler

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        return running_loss / len(val_loader)
    

    def train(self, train_loader, val_loader, epochs):
        """
        main.py에서의 학습 루프를 이 함수로 옮김
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
            
            if self.best_val_loss > val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model.state_dict()
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Best model saved with Validation Loss: {val_loss}")

            # 조기 종료 로직
            if self.early_stop and self.patience_limit is not None:
                if val_loss >= self.best_val_loss:
                    self.patience_check += 1
                    print(f"Early stopping patience: {self.patience_check}/{self.patience_limit}")
                    if self.patience_check >= self.patience_limit:
                        print("Early stopping triggered.")
                        break
                else:
                    self.patience_check = 0
            
            self.scheduler.step(val_loss)

