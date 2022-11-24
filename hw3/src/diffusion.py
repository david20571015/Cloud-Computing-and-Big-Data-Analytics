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

        x_t = (self.sqrt_alpha_bar[times] * x_0 +
               self.sqrt_one_minus_alpha_bar[times] * noise)

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
        x_t = x_t - self.beta[t] / self.sqrt_one_minus_alpha_bar[t] * pred_noise
        x_t = x_t / self.sqrt_alpha[t]
        return x_t

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        x_t = x_T

        # In order to match the index of time embedding, set time from T-1 to 0.
        for times in reversed(torch.arange(self.time_steps, device=x_T.device)):
            z = torch.randn_like(x_t) if times > 0 else 0
            x_t = self.predict_mean(x_t, times) + self.sigma[times] * z
        return x_t.clip(-1.0, 1.0)

    @torch.no_grad()
    def grid_sample(self, x_T: torch.Tensor, n_rows=8) -> torch.Tensor:
        save_index = torch.linspace(0,
                                    self.time_steps - 1,
                                    n_rows,
                                    dtype=torch.long,
                                    device=x_T.device)[1:]
        save_images = [torch.zeros_like(x_T)]
        x_t = x_T

        # In order to match the index of time embedding, set time from T-1 to 0.
        for times in reversed(torch.arange(self.time_steps, device=x_T.device)):
            z = torch.randn_like(x_t) if times > 0 else 0
            x_t = self.predict_mean(x_t, times) + self.sigma[times] * z
            if times in save_index:
                save_images.append(x_t.clip(-1., 1.))
        return torch.cat(save_images, dim=0)


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
