import torch
from torch import nn
from torch.nn import functional as F


class TimestepEmbedding(nn.Module):

    def __init__(self, time_steps, emb_dim) -> None:
        super().__init__()

        pos = torch.arange(time_steps)
        log_emb = -torch.arange(0, emb_dim, 2) / emb_dim * torch.log(
            torch.tensor(10000.0))
        emb = torch.exp(log_emb[None, :]) * pos[:, None]
        emb = torch.concat([emb.sin(), emb.cos()], dim=-1)

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
        )

    def forward(self, x: torch.IntTensor | torch.LongTensor) -> torch.Tensor:
        return self.embedding(x)


class Upsample(nn.Module):

    def __init__(self, ch: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, *_args) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, ch: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, *_args) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResBlock(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.pos_emb_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim * 4, out_ch),
        )
        self.conv_block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        y = self.conv_block1(x)
        y += self.pos_emb_block(pos_emb)[..., None, None]
        y = self.conv_block2(y)
        return y + self.shortcut(x)


class SelfAttentionBlock(nn.Module):

    def __init__(self, ch) -> None:
        super().__init__()

        self.normalize = nn.GroupNorm(32, ch)
        self.query = nn.Conv2d(ch, ch, kernel_size=1)
        self.key = nn.Conv2d(ch, ch, kernel_size=1)
        self.value = nn.Conv2d(ch, ch, kernel_size=1)
        self.out = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor, *_args) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.normalize(x)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        q = q.permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = k.reshape(B, C, H * W)
        alpha = torch.bmm(q, k) / (C**0.5)
        alpha = F.softmax(alpha, dim=-1)

        v = v.permute(0, 2, 3, 1).reshape(B, H * W, C)
        b = torch.bmm(alpha, v).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.out(b) + x


class UNet(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        time_steps: int,
        channels: int,
        num_res_blocks: int,
        ch_mults=(1, 2, 4),
        attn_res=(16, 8, 4)) -> None:
        super().__init__()

        if input_shape[1] != input_shape[2]:
            raise ValueError(
                f'input_shape must be square, got: "{input_shape[1:]}".')
        if input_shape[1] % 2**len(ch_mults) != 0:
            raise ValueError(
                f'w and h of input_shape must be divisible by 2**{len(ch_mults)},'
                f' got: "{input_shape[1:]}".')

        self.time_embedding = TimestepEmbedding(time_steps, channels)
        self.stem = nn.Conv2d(input_shape[0],
                              channels,
                              kernel_size=3,
                              padding=1)

        self.down_blocks = nn.ModuleList()
        curr_res = input_shape[1]
        curr_ch = channels
        ch_buffer = [channels]
        for i, ch_mult in enumerate(ch_mults):
            out_ch = channels * ch_mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(in_ch=curr_ch,
                             out_ch=out_ch,
                             emb_dim=channels,
                             dropout=0.1))
                curr_ch = out_ch
                ch_buffer.append(curr_ch)

                if curr_res in attn_res:
                    self.down_blocks.append(SelfAttentionBlock(curr_ch))

            if i != len(ch_mults) - 1:
                self.down_blocks.append(Downsample(ch=curr_ch))
                ch_buffer.append(curr_ch)
                curr_res //= 2

        self.middle_block = nn.ModuleList([
            ResBlock(in_ch=curr_ch,
                     out_ch=curr_ch,
                     emb_dim=channels,
                     dropout=0.1),
            SelfAttentionBlock(ch=curr_ch),
            ResBlock(in_ch=curr_ch,
                     out_ch=curr_ch,
                     emb_dim=channels,
                     dropout=0.1),
        ])

        self.up_blocks = nn.ModuleList()
        for i, ch_mult in enumerate(reversed(ch_mults)):
            out_ch = channels * ch_mult

            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(in_ch=curr_ch + ch_buffer.pop(),
                             out_ch=out_ch,
                             emb_dim=channels,
                             dropout=0.1))
                curr_ch = out_ch

                if curr_res in attn_res:
                    self.up_blocks.append(SelfAttentionBlock(curr_ch))

            if i != len(ch_mults) - 1:
                self.up_blocks.append(Upsample(ch=curr_ch))
                curr_res *= 2

        self.out_block = nn.Sequential(
            nn.GroupNorm(32, curr_ch),
            nn.SiLU(),
            nn.Conv2d(curr_ch, input_shape[0], kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        emb = self.time_embedding(time)
        x = self.stem(x)

        xs = [x]
        for block in self.down_blocks:
            x = block(x, emb)
            if not isinstance(block, SelfAttentionBlock):
                xs.append(x)

        for block in self.middle_block:
            x = block(x, emb)

        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                x = torch.cat([x, xs.pop()], dim=1)
            x = block(x, emb)

        x = self.out_block(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 28, 28)
    times = torch.arange(1)

    model = UNet(input_shape=(3, 28, 28),
                 time_steps=10,
                 channels=32,
                 num_res_blocks=4)

    print(model(inputs, times).shape)  # torch.Size([1, 3, 28, 28])
