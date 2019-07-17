import time
import torch
from torch import nn, optim
import torch.nn.functional as F

print('Device: ', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)                        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
        self.criterion = nn.NLLLoss()
        
        # Use GPU if it's available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device);
        
    def get_device(self):
        return "cuda" if next(self.parameters()).is_cuda else "cpu"
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))        
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
    
    def process_batch(self, inputs, labels):
        model = self
        #print('  batch: ', batch_number)
        # start = self.log_time_and_reset(start, "    Get batch:", profile)
        # batch_number += 1                
        # start = time.time() #Added by Edgarin
        
        # start = self.log_time_and_reset(start, "    Move to device:", profile)        

        log_ps = model.forward(inputs)
        # start = self.log_time_and_reset(start, "    model.forward(): ", profile)
        loss = self.criterion(log_ps, labels)
        return (log_ps, loss)
    
    def train_dataset(self, trainloader):
        model = self
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        train_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(self.get_device()), labels.to(self.get_device()) # Added by Edgarin
            optimizer.zero_grad()
            (log_ps, loss) = self.process_batch(images, labels)
            loss.backward()
            optimizer.step()
            # start = self.log_time_and_reset(start, "    backward and step: ", profile)
            train_loss += loss.item()
        return (train_loss)
    
    def predict_dataset(self, testloader, isValidation = False, return_predictions = False):
        model = self
        test_loss = 0
        accuracy = 0
        
        predictions = torch.tensor([]).long()

        # Turn off gradients for validation, saves memory and computations                
        with torch.no_grad():            
            if isValidation: model.eval()  # To activate dropouts
            for images, labels in testloader:
                #start = time.time() #Added by Edgarin
                images, labels = images.to(self.get_device()), labels.to(self.get_device()) # Added by Edgarin
                (log_ps, loss) = self.process_batch(images, labels)
                test_loss += loss

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                #print(f"Device = {self.get_device()}; Last test batch time: {(time.time() - start):.4f} seconds") #Added by Edgarin
                
                if return_predictions:
                    prediction = top_class.view(top_class.shape[0]).cpu()
                    predictions = torch.cat((predictions, prediction), 0)
            else:
                model.train()        
        return (accuracy, test_loss, predictions if return_predictions else None)
    
  
    def log_time_and_reset(self, start, message = "{:.4f} seconds", profile = False):
        new_start = time.time()
        if profile: print((message + " {:.0f} ms").format((new_start - start)*1000)) #Added by Edgarin
        return new_start
    
    
    def train_and_test(self, trainloader, testloader, epochs = 1, profile = False):        
        model = self                        

        # epochs = 30  #Only thing changed from original
        steps = 0

        train_losses, test_losses = [], []
        for e in range(epochs):
            print('Epoch {}'.format(e+1))
            b = 0
            epoch_start = time.time()
            start = time.time()
            (train_loss) = self.train_dataset(trainloader)
                        
            #else:
            if profile: print(f"  Epoch time: {(time.time() - epoch_start):.4f} seconds") #Added by Edgarin
            if(True):                
                ###
                (accuracy, test_loss, _) = self.predict_dataset(testloader, isValidation = True)
                
                train_losses.append(train_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))

                print("Epoch: {}/{} ".format(e+1, epochs),
                      "Training Loss: {:.3f} ".format(train_loss/len(trainloader)),
                      "Test Loss: {:.3f} ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    