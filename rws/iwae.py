import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import BasicModel

def train_iwae(model, k, train_data, valid_data, n_epochs, init_lr, batch_size):

	optimizer = optim.Adam(model.parameters(), lr=init_lr)
	for z in range(n_epochs):
		n_subep = int(np.power(3, z))
		lr = lr_init * np.power(10, -z / 7.0)
		optimizer.lr = lr
		for w in range(n_subep):
			n_batches = int(train_data.shape[0] / batch_size)
			for q in range(n_batches):
				ind = np.random.choice(train_data.shape[0], batch_size, replace=False)
				batch = np.zeros((k * batch_size, train_data.shape[1]))

				for i in range(batch.shape[0]):
					wh = int(i / k)
					batch[i] = train_data[ind[wh]]

				batch = torch.FloatTensor(batch)
				sample, mu, sigma, output = model(batch, reparameterization=True)

				