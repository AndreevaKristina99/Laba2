class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 7, stride = 3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features =32),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer02 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 3,
                               stride = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.Dropout(0.2)
                )
        self.layer01 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size = 7, stride
            = 3),
            nn.Sigmoid()
        )
        def forward(self, x):
           out = self.layer1(x)
           out = self.layer2(out)
           out = self.layer02(out)
           out = self.layer01(out)
           return out
        if torch.cuda.is_available() == True:
           device = "cuda:0"
        else:
            device = "cpu"
criterion = nn.MSELoss()
optimizer = optim. Adam(model.parameters(), lr = 0.003)
epochs = 120
l = len(trainloader)
losslist = list()
running_loss = 0
KOst = 0.0063
for epoch in range(epochs):
    print("Entering Epoch: ",epoch)
    for i1, (dirty, clean, label) in tqdm (enumerate
                                           (trainloader)):
        dirty = dirty.view(dirty.size(0),1,28,28).type(torch.FloatTensor)
        clean = clean.view(clean.size(0),1,28,28).type(torch.FloatTensor)
        dirty, clean = dirty.to(device),clean.to(device)
        output = model.forward(dirty)
        loss = criterion(output,clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epochloss += loss.item()
        sr_loss = running_loss/l
        losslist.append(sr_loss)
        running_loss=0
        print("======> epoch: {}/{}, Loss:{}".format(epoch,
                                                     epochs, sr_loss))
        if sr_loss <= KOst: break








