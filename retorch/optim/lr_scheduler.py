from .sgd import SGD

class StepLR():
    def __init__(self, optimizer: SGD, step_size: int, gamma: float) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        
        if (self._step_count % self.step_size) == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma