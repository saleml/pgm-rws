import torch
import torch.nn as nn
torch.random.manual_seed(10)

class model(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(10, 10)

	def forward(self, x):
		return self.fc1(x)

net = model()
x = torch.ones(10, 10)
net.zero_grad()
out = net(x)
to = torch.sum(out, dim=0)
#to = torch.zeros(10)
#for i in range(10):
#	to = to + out[i]
loss = to.sum()
print(loss)
loss.backward()

for param in net.parameters():
	print(param.grad)