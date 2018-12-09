import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


class BaseAlgo:
    def __init__(self, model, mode='MNIST'):
        self.model = model
        self.mode = mode
        self.train_loss = 0.

    def test_step(self, data):
        raise NotImplementedError("Implemented in the subclasses")

    def test_log(self, epoch, test_loader, batch_size, step=None):
        if self.mode == 'dis-GMM':
            if step % 100 == 0:
                samples = self.model.sample(1000).numpy()
                test_samples = test_loader.next_batch(1000)

                plt.scatter(samples[:, 0], samples[:, 1], c='b', label='Samples from learned model')
                plt.scatter(test_samples[:, 0], test_samples[:, 1], c='r', label='Actual test samples')
                plt.savefig('results_GMM/sample_' + str(step) + '.png')
                plt.legend()
                plt.gcf().clear()

        if self.mode == 'MNIST':
            test_loss = 0.
            for batch_idx, (data, target) in enumerate(test_loader):
                _, _, recon, loss = self.test_step(data)

                if isinstance(loss, tuple):
                    sum_loss = sum(list(loss))
                else:
                    sum_loss = loss

                test_loss += sum_loss.item() * batch_size

                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data.view(batch_size, 1, 28, 28)[:n],
                                            recon.view(batch_size, 1, 28, 28)[:n]])
                    save_image(comparison,
                               './results_MNIST/reconstruction_' + str(epoch) + '.png', nrow=n)

            print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))

            with torch.no_grad():
                samples = self.model.sample(64)
                save_image(samples.view(64, 1, 28, 28),
                           'results_MNIST/sample_' + str(epoch) + '.png')

    def visu(self, writer, step, args, epoch=None, batch_idx=None,
             len_data=None, len_whole_data=None, batch_size=None,
             latent_proba=None, mus=None):
        mean, logvar, loss = args

        if isinstance(loss, tuple):
            sum_loss = sum(list(loss))
        else:
            sum_loss = loss

        if self.mode == 'dis-GMM':
            if step % 100 == 0:
                pi_diff = np.sum(np.abs(np.sort(self.model.pi.detach().numpy()) - np.sort(latent_proba)))
                mu_pred = self.model.mus.detach().numpy()
                mu_diff = np.sum(np.abs(mu_pred[mu_pred[:, 0].argsort()] - mus[mus[:, 0].argsort()]))
                print('====> Step: {} Average loss: {:.4f}, L1_pi: {:.7f}, L1_mu {:.7f}'.format(
                    step, sum_loss.item(), pi_diff, mu_diff))
                writer.add_scalar('L1_pi', pi_diff, step)
                writer.add_scalar('L1_mu', mu_diff, step)

                if isinstance(loss, tuple):
                    loss_model, loss_q_wake, loss_q_sleep = loss
                    writer.add_scalar('model_loss', loss_model.item(), step)
                    writer.add_scalar('recon_sleep_loss', loss_q_sleep.item(), step)
                    writer.add_scalar('recon_wake_loss', loss_q_wake.item(), step)
                else:
                    writer.add_scalar('loss', loss, step)

        if self.mode == 'MNIST':
            self.train_loss += sum_loss.item() * batch_size

            if (batch_idx + 1) * batch_size >= len_whole_data:
                # last step of the epoch
                print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, self.train_loss / len_whole_data))

            if (batch_idx % 100) == 0:
                num_batches = len_whole_data / batch_size
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len_data, len_whole_data,
                                                                               100. * batch_idx / num_batches,
                                                                               sum_loss.item()))

                if isinstance(loss, tuple):
                    loss_model, loss_q_wake, loss_q_sleep = loss
                    writer.add_scalar('model_loss', loss_model.item(), step)
                    writer.add_scalar('recon_sleep_loss', loss_q_sleep.item(), step)
                    writer.add_scalar('recon_wake_loss', loss_q_wake.item(), step)
                else:
                    writer.add_scalar('loss', loss, step)

                eps = torch.rand_like(mean)
                h = mean + eps * torch.exp(logvar / 2)
                out, _, _ = self.model.decode(eps)
                out = out.view(mean.size()[0], 1, 28, 28)
                rec, _, _ = self.model.decode(h)
                rec = rec.view(mean.size()[0], 1, 28, 28)
                writer.add_image('im_0', out[0], step)
                writer.add_image('im_1', out[1], step)
                writer.add_image('rec_0', rec[2], step)
                writer.add_image('rec_1', rec[3], step)