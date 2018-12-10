import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from data.utils import L2_difference_perm


class BaseAlgo:
    def __init__(self, model, mode='MNIST'):
        self.input_dim = None
        self.model = model
        self.mode = mode
        self.train_loss = 0.
        self.scheduler, self.scheduler_model = None, None

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
             latent_proba=None, mus=None, test_set=None, test_posteriors=None, csv_writer=None, path=None):
        mean, logvar, loss = args

        if isinstance(loss, tuple):
            sum_loss = sum(list(loss))
        else:
            sum_loss = loss

        if self.mode == 'dis-GMM':
            if step % 100 == 0:
                mu_pred = self.model.mus.detach().numpy()
                mu_diff, best_perm = L2_difference_perm(mus, mu_pred, self.model.encoding_dim)
                pi_diff = np.sqrt(np.sum((self.model.pi.detach().numpy()[best_perm] - latent_proba) ** 2))

                _, posteriors, _ = self.model.encode(test_set)
                L2_posteriors = np.mean(np.sqrt(np.sum((posteriors.detach().numpy() - test_posteriors) ** 2, axis=1)))

                print('====> Step: {} Average loss: {:.4f}, L2_pi: {:.4f}, L2_mu: {:.3f}, L2_posteriors: {:.3f}'.format(
                    step, sum_loss.item(), pi_diff, mu_diff, L2_posteriors))
                writer.add_scalar('L2_pi', pi_diff, step)
                writer.add_scalar('L2_mu', mu_diff, step)
                writer.add_scalar('L2_posteriors', L2_posteriors, step)

                data = [step, pi_diff, mu_diff, L2_posteriors]

                if isinstance(loss, tuple):
                    loss_model, loss_q_wake, loss_q_sleep = loss
                    writer.add_scalar('model_loss', loss_model.item(), step)
                    writer.add_scalar('recon_sleep_loss', loss_q_sleep.item(), step)
                    writer.add_scalar('recon_wake_loss', loss_q_wake.item(), step)
                    scheduler = self.scheduler_model
                    data += [loss_model.item(), loss_q_sleep.item(), loss_q_wake.item()]
                else:
                    writer.add_scalar('loss', loss.item(), step)
                    scheduler = self.scheduler
                    data += [loss.item()]

                if scheduler is not None:
                    writer.add_scalar('lr', scheduler.get_lr()[0], step)

                csv_writer.writerow(data)

                if step % 3000 == 0:
                    torch.save(self.model, path + '/model_step_{}.pt'.format(step))

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
                    scheduler = self.scheduler_model
                else:
                    writer.add_scalar('loss', loss, step)
                    scheduler = self.scheduler

                if scheduler is not None:
                    writer.add_scalar('lr', scheduler.get_lr()[0], step)

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