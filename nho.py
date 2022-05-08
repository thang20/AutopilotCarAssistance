import torchvision.models as models # them model
import torch.nn as nn
import torch
model = models.resnet18()
print(model)
a = nn.Sequential(*list(model.children())[0:-1])
img = np.transpose(img, (2, 0, 1))# doi chieu img
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get data to cuda if possible
data = data.to(device=device)
targets = targets.to(device=device)

# forward
scores = model(data)
loss = criterion(scores, targets)
loss_value = loss.data.cpu().numpy()
batch_losses.append(loss_value)

# backward
optimizer.zero_grad()
loss.backward()

# gradient descent or adam step
optimizer.step()



