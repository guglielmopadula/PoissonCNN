import numpy as np

from tqdm import trange
from torch import nn
import torch
import matplotlib.pyplot as plt




import matplotlib.pyplot as plt
import numpy as np
from torch import nn
inputs=np.load("inputs.npy")
outputs=np.load("outputs.npy")


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.hidden_dim=50
        self.nn=nn.Sequential(nn.ConvTranspose2d(1,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,self.hidden_dim,2,2),nn.ReLU(),nn.Dropout(0.05),
                              nn.ConvTranspose2d(self.hidden_dim,1,2,2))


        

        

    def forward(self,x):
        x=x.reshape(x.shape[0],1,1,1)
        return self.nn(x)
    


BATCH_SIZE=20
TRAIN_SIZE=80
TEST_SIZE=20
inputs=torch.tensor(np.load("inputs.npy")).float()
outputs=torch.tensor(np.load("outputs.npy")).float()
inputs_train=inputs[:TRAIN_SIZE]
inputs_test=inputs[TRAIN_SIZE:] 
outputs_train=outputs[:TRAIN_SIZE]
outputs_test=outputs[TRAIN_SIZE:]
dataset_train=torch.utils.data.TensorDataset(inputs_train,outputs_train)
dataloader_train=torch.utils.data.DataLoader(dataset_train,batch_size=BATCH_SIZE)
dataset_test=torch.utils.data.TensorDataset(inputs_test,outputs_test)
dataloader_test=torch.utils.data.DataLoader(dataset_test,batch_size=BATCH_SIZE)

model = CNN()
n_epochs = 500
lr=0.01

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in trange(0, n_epochs):
    for data in dataloader_train:
        inputs,outputs=data
        optimizer.zero_grad()
        outputs_hat = model(inputs).reshape(outputs.shape)
        loss = criterion(outputs_hat,outputs)
        loss.backward() 
        optimizer.step()
    print(torch.linalg.norm(outputs_hat.reshape(BATCH_SIZE,-1)-outputs.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs.reshape(BATCH_SIZE,-1)))

for data in dataloader_test:
    inputs,outputs=data
    outputs_hat = model(inputs).reshape(outputs.shape)
    print(torch.linalg.norm(outputs_hat.reshape(BATCH_SIZE,-1)-outputs.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs.reshape(BATCH_SIZE,-1)))