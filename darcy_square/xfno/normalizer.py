class Normalizer():
    def __init__(self, x, eps=1e-5):
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.eps = eps
        
    def encode(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = x * (self.std + self.eps) + self.mean
        return x