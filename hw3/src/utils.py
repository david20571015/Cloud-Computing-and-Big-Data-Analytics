from torch import nn

from src.models import UNet


class ExponentialMovingAverage:

    def __init__(
        self,
        source: nn.Module,
        target: nn.Module,
        decay=0.999,
    ) -> None:
        self.source = source
        self.target = target
        self.decay = decay

    def step(self) -> None:
        for source_param, target_param in zip(self.source.parameters(),
                                              self.target.parameters()):
            target_param.data.copy_(self.decay * target_param.data +
                                    (1.0 - self.decay) * source_param.data)


if __name__ == '__main__':
    config = {
        'input_shape': (3, 32, 32),
        'time_steps': 10,
        'channels': 32,
        'num_res_blocks': 2
    }
    src = UNet(**config)
    tar = UNet(**config)
    print(id(src), id(tar))

    ema = ExponentialMovingAverage(src, tar)
    print(id(ema.source), id(ema.target))
