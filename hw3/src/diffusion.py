from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from src.models import UNet


class DiffusionTrainer(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        time_steps=1000,
        beta_1=1e-4,
        beta_T=2e-2,
    ):
        super().__init__()
        self.model = model
        self.time_steps = time_steps

        beta = torch.linspace(beta_1, beta_T, time_steps,
                              dtype=torch.float32).view(-1, 1, 1, 1)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar',
                             torch.sqrt(1.0 - alpha_bar))

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        times = torch.randint(self.time_steps,
                              size=(x_0.shape[0],),
                              device=x_0.device)
        noise = torch.randn_like(x_0)

        x_t = (self.get_buffer('sqrt_alpha_bar')[times] * x_0 +
               self.get_buffer('sqrt_one_minus_alpha_bar')[times] * noise)

        pred_noise = self.model(x_t, times)
        loss = F.mse_loss(pred_noise, noise)
        return loss


class DiffusionSampler(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        time_steps=1000,
        beta_1=1e-4,
        beta_T=2e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps

        beta = torch.linspace(beta_1, beta_T, time_steps,
                              dtype=torch.float32).view(-1, 1, 1, 1)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('sqrt_alpha', torch.sqrt(alpha))
        self.register_buffer('sqrt_one_minus_alpha_bar',
                             torch.sqrt(1.0 - alpha_bar))
        self.register_buffer('sigma', torch.sqrt(beta))

    def predict_mean(self, x_t, t) -> torch.Tensor:
        pred_noise = self.model(x_t, t)
        x_t = x_t - self.get_buffer('beta')[t] / self.get_buffer(
            'sqrt_one_minus_alpha_bar')[t] * pred_noise
        x_t = x_t / self.get_buffer('sqrt_alpha')[t]
        return x_t

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        x_t = x_T

        # In order to match the index of time embedding, set time from T-1 to 0.
        for t in reversed(torch.arange(self.time_steps, device=x_T.device)):
            z = torch.randn_like(x_t) if t > 0 else 0
            x_t = self.predict_mean(x_t, t) + self.get_buffer('sigma')[t] * z
        return x_t.clip(-1.0, 1.0)

    @torch.no_grad()
    def grid_sample(self, x_T: torch.Tensor, n_rows=8) -> torch.Tensor:
        save_index = torch.linspace(0,
                                    self.time_steps,
                                    n_rows,
                                    dtype=torch.long,
                                    device=x_T.device)[:-1]  # exclude T-1
        save_images = [torch.zeros_like(x_T)]
        x_t = x_T

        # In order to match the index of time embedding, set time from T-1 to 0.
        for t in reversed(torch.arange(self.time_steps, device=x_T.device)):
            z = torch.randn_like(x_t) if t > 0 else 0
            x_t = self.predict_mean(x_t, t) + self.get_buffer('sigma')[t] * z
            if t in save_index:
                save_images.append(x_t.clip(-1., 1.))
        return torch.cat(save_images, dim=0)


class DdimSampler(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        time_steps=1000,
        sample_steps=50,
        step_type: Literal['linear', 'quadratic'] = 'linear',
        beta_1=1e-4,
        beta_T=2e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.sample_steps = sample_steps

        if step_type == 'linear':
            tau = torch.linspace(0,
                                 time_steps - 1,
                                 sample_steps,
                                 dtype=torch.long)
        elif step_type == 'quadratic':
            tau = torch.arange(sample_steps)**2 * \
                    ((time_steps - 1) / (sample_steps - 1)**2)
            tau = tau.floor().long()
        else:
            raise KeyError(
                f'step_type should be "linear" or "quadratic", but got {step_type}'
            )

        self.register_buffer('tau', tau)

        beta = torch.linspace(beta_1, beta_T, time_steps,
                              dtype=torch.float32).view(-1, 1, 1, 1)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('alpha_bar', alpha_bar[tau])

        # Compute sigma[1:] first because alpha_bar_0 is specified as 1.
        sigma = torch.sqrt((1 - alpha_bar[tau][:-1]) / (1 - alpha_bar[tau][1:]))
        sigma = sigma * torch.sqrt(1 - alpha[tau][1:])
        sigma = torch.cat([torch.zeros_like(sigma[0])[None, :], sigma], dim=0)

        self.register_buffer('sigma', sigma)

    def forward(self, x_T: torch.Tensor, eta=0.0) -> torch.Tensor:
        x_t = x_T
        sigma = self.get_buffer('sigma') * eta

        for t in reversed(
                torch.arange(self.sample_steps, device=x_T.device)[1:]):
            pred_noise = self.model(x_t, self.get_buffer('tau')[t])

            pred_x_0 = x_t - torch.sqrt(
                1 - self.get_buffer('alpha_bar')[t]) * pred_noise
            pred_x_0 = pred_x_0 / torch.sqrt(self.get_buffer('alpha_bar')[t])

            dir_x_t = torch.sqrt(1 - self.get_buffer('alpha_bar')[t - 1] -
                                 sigma[t]**2) * pred_noise

            random_noise = sigma[t] * torch.randn_like(x_t)

            x_t = pred_x_0 * self.get_buffer('alpha_bar')[
                t - 1].sqrt() + dir_x_t + random_noise

        pred_noise = self.model(x_t, torch.tensor(0, device=x_T.device))
        pred_x_0 = x_t - torch.sqrt(
            1 - self.get_buffer('alpha_bar')[0]) * pred_noise
        x_0 = pred_x_0 / torch.sqrt(self.get_buffer('alpha_bar')[0])

        return x_0.clip(-1.0, 1.0)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 32, 32)
    backbone = UNet(input_shape=input_shape,
                    time_steps=10,
                    channels=32,
                    num_res_blocks=2)
    trainer = DiffusionTrainer(backbone, time_steps=10).to(DEVICE)
    print(trainer(torch.randn(8, *input_shape, device=DEVICE)))

    sampler = DiffusionSampler(backbone, time_steps=10).to(DEVICE)
    print(sampler(torch.randn(8, *input_shape, device=DEVICE)).shape)

    sampler = DdimSampler(backbone, time_steps=1000).to(DEVICE)
    print(sampler(torch.randn(8, *input_shape, device=DEVICE)).shape)
