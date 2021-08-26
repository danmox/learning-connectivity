from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid


class View(nn.Module):
    """helper class to provide view functionality in sequential containers"""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AEBase(pl.LightningModule):
    """base class for connectivity autoencoder models"""

    def __init__(self, log_step):
        super().__init__()

        # cache some data to show learning progress
        self.train_progress_batch = None
        self.val_progress_batch = None

        # want logging frequency less than every training iteration and more
        # than every epoch
        self.log_step = log_step
        self.loss_hist = 0.0
        self.log_it = 0

        # keep track of the best validation models
        self.val_loss_best = float('Inf')
        self.best_model_path = None

        self.model_name = ''

        # initialize network
        self.init_model()
        self.init_weights()

    def init_model():
        """implemented by derived classes"""
        raise NotImplementedError()

    def init_weights(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    # log network output for a single batch
    def log_network_image(self, batch, name):
        x, y = batch
        y_hat = self.output(x)

        img_list = []
        for i in range(x.shape[0]):
            img_list.append(x[i,...].cpu().detach())
            img_list.append(y_hat[i,...].cpu().detach())
            img_list.append(y[i,...].cpu().detach())

        grid = torch.clamp(make_grid(img_list, nrow=3, padding=20, pad_value=1), 0.0, 1.0)
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def _ckpt_dir(self):
        log_dir = Path(self.trainer.weights_save_path) / self.logger.save_dir
        return log_dir / f'version_{self.logger.version}' / 'checkpoints'

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss

    # provide visual feedback of the learning progress after every epoch
    def training_epoch_end(self, outs):
        torch.set_grad_enabled(False)
        self.eval()

        self.log_network_image(self.train_progress_batch, 'train_results')

        torch.set_grad_enabled(True)
        self.train()

    def validation_step(self, batch, batch_idx):
        # set aside some data to show network performance on validation data
        if self.val_progress_batch is None:
            self.val_progress_batch = batch

        x, y = batch
        y_hat = self.output(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_epoch_end(self, outs):
        val_loss = np.mean(np.asarray([o.item() for o in outs]))

        # save checkpoint of best performing model
        if val_loss < self.val_loss_best:
            self.val_loss_best = val_loss

            if self.best_model_path is not None:
                self.best_model_path.unlink(missing_ok=True)

            filename = self._ckpt_dir() / (self.model_name + f'valloss_{val_loss:.3e}_epoch_{self.current_epoch}.ckpt')
            self.trainer.save_checkpoint(filename, weights_only=True)
            self.best_model_path = filename

        self.logger.experiment.add_scalar('val_loss', val_loss, self.current_epoch)
        self.log_network_image(self.val_progress_batch, 'val_results')

    def output(self, x):
        """like self.forward but only returns the image without training extras"""
        return self(x)

    def evaluate(self, x):
        """run inference on the network given an tensor"""
        with torch.no_grad():
            x = x[None, None] / 255.0
            y_hat = self.output(x.float())
            y_hat = torch.clamp(255*y_hat, 0, 255).to(torch.uint8).squeeze()
        return y_hat

    def inference(self, x):
        """run inference on the network given a numpy array"""
        return self.evaluate(torch.from_numpy(x)).cpu().detach().numpy()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class UAEBase(AEBase):
    """your stock undercomplete auto encoder"""

    def __init__(self, log_step=1):
        super().__init__(log_step)

    def forward(self, x):
        return self.network(x)


class BetaVAEBase(AEBase):
    """Beta Variational Auto Encoder class for learning connectivity from images

    based on BetaVAE_B from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    which is based on: https://arxiv.org/abs/1804.03599

    """

    def __init__(self, log_step=1):
        super().__init__(log_step)

    def forward(self, x):

        # encoder spits out latent distribution as a single 32x1 vector with the
        # first 16 elements corresponding to the mean and the last 16 elements
        # corresponding to the log of the variance
        latent_distribution = self.encoder(x)
        mu = latent_distribution[:, :self.z_dim]
        logvar = latent_distribution[:, self.z_dim:]

        if self.training:
            # generate the input to the decoder using the reparameterization trick
            std = logvar.div(2).exp()
            eps = torch.randn_like(std)
            z = mu + std*eps
        else:
            z = mu

        out = self.decoder(z)

        return out, mu, logvar

    def output(self, x):
        return self(x)[0]

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress on training data
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + self.beta * self.kld_weight * kld_loss

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss
