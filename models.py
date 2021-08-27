import torch.nn as nn
from models_base import UAEBase, BetaVAEBase, View


#
# UAEModels
#

class UAEModel_256_nf32_8x8kern_fc256(UAEBase):

    def __init__(self, log_step=1, **kwargs):
        super().__init__(log_step)
        self.model_name = self.__class__.__name__

    def init_model(self):
        nf = 32
        self.network = nn.Sequential(            #  1, 256, 256 (input)
            nn.Conv2d(1, nf, 8, 2, 3),           # nf, 128, 128
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 8, 2, 3),          # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 8, 2, 3), # nf, 128, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 8, 2, 3)   # nf, 256, 256
        )                                        #  1, 256, 256 (output)


#
# BetaVAEModels
#


class BetaVAEModel_128_b1_z16_nf32_4x4kern(BetaVAEBase):

    def __init__(self, kld_weight=1, log_step=1):
        super().__init__(log_step)
        self.kld_weight = kld_weight
        self.model_name = self.__class__.__name__

    def init_model(self):
        self.beta = 1.0
        self.z_dim = 16 # dimension of latent distribution

        nf = 32
        self.encoder = nn.Sequential(            #  1, 128, 128 (input)
            nn.Conv2d(1, nf, 4, 2, 1),           # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 2*self.z_dim)         # 2 x z_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),          # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 4, 2, 1)   # nf, 128, 128
        )                                        #  1, 128, 128 (output)


class BetaVAEModel_256_b1_z16_nf48_8x8kern(BetaVAEBase):

    def __init__(self, kld_weight=1, log_step=1):
        super().__init__(log_step)
        self.kld_weight = kld_weight
        self.model_name = self.__class__.__name__

    def init_model(self):
        self.beta = 1.0
        self.z_dim = 16 # dimension of latent distribution

        nf = 48
        self.encoder = nn.Sequential(            #  1, 256, 256 (input)
            nn.Conv2d(1, nf, 8, 2, 3),           # nf, 128, 128
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 8, 2, 3),          # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 2*self.z_dim)         # 2 x z_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),          # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 8, 2, 3), # nf, 128, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 8, 2, 3)   # nf, 256, 256
        )                                        #  1, 256, 256 (output)


class BetaVAEModel_256_b1_z16_nf32_4x4kern(BetaVAEBase):

    def __init__(self, kld_weight=1, log_step=1):
        super().__init__(log_step)
        self.kld_weight = kld_weight
        self.model_name = self.__class__.__name__

    def init_model(self):
        self.beta = 1.0
        self.z_dim = 16 # dimension of latent distribution

        nf = 32
        self.encoder = nn.Sequential(            #  1, 256, 256 (input)
            nn.Conv2d(1, nf, 4, 2, 1),           # nf, 128, 128
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 2*self.z_dim)         # 2 x z_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),          # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf, 128, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 4, 2, 1)   # nf, 256, 256
        )                                        #  1, 256, 256 (output)


#
# ConvAEModels
#


class ConvAEModel_nf64_8x8kern(UAEBase):
    def __init__(self, log_step=1, **kwargs):
        super().__init__(log_step)
        self.model_name = self.__class__.__name__

    def init_model(self):
        nf = 64
        self.network = nn.Sequential(            #  1, 128, 128 (input)
            nn.Conv2d(1, nf, 8, 2, 3),           # nf,  64,  64
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 8, 2, 3),          # nf,  32,  32
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 8, 2, 3), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 8, 2, 3),  # nf, 128, 128
        )                                        #  1, 128, 128 (output)
