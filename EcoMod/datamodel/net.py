from numpy import linspace

class Net:
    def __init__(self, tmin, tmax, N):
        assert tmin < tmax
        self.tmin = tmin
        self.tmax = tmax
        self.N = N
        self.h = (tmax - tmin) / N

    def to_argument(self, i):
        assert 0 < i and i < self.N + 1
        return self.tmin + self.h * (i - 1)

    def to_point(self, t):
        assert self.tmin <= t and t <= self.tmax
        return int(1 + (t - self.tmin) / self.h)

    def as_array(self):
        return linspace(self.tmin, self.tmax, self.N)