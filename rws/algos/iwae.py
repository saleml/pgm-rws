import torch


class IWAE:
    def __init__(self, model, data_loader, optimizer, num_epochs, save_interval=10):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer

        self.losses = []

        self.num_epochs = num_epochs
        self.save_interval = save_interval

    def loss(self, inputs):
        """Define the loss"""
        pass

    def run_epoch(self):
        """Loop through the data and make gradient updates (populate the losses variable)"""
        pass

    def train(self):
        for epoch in range(self.num_epochs):
            self.run_epoch()
            if (epoch + 1) % self.save_interval == 0:
                torch.save(self.model.state_dict(), "IWAE_model_epoch_{}.pt".format(epoch))