class StepDecay:
    """
    This class gives a learning rate based on the epoch. You can choose to increase or decrease the learning rate, throughout the training, 
    allowing for better training results.
    """
    def __init__(self, initial_lr, drop_every, drop_ratio):
        self.initial_lr = initial_lr
        self.drop_every = drop_every
        self.drop_ratio = drop_ratio

    def __call__(self, epoch):
        return self.initial_lr * (self.drop_ratio ** (epoch // self.drop_every))