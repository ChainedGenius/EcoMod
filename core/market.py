class Flow(object):
    """
        Here we will contain parsing results from flows file?
    """

    def __init__(self, producer, receiver, value, dim):
        self.producer = producer
        self.receiver = receiver
        self.value = value
        self.dim = dim
        pass

    def __str__(self):
        return f'{self.producer.name} ---- {self.value} ---> {self.receiver.name}'


class Market(object):
    def __init__(self, eq, dim, lagents):
        self.eq = eq
        self.dim = dim
        self.lagents = lagents
        pass


class MarketValidator(object):
    def __market_closureness(self, markets):
        pass

    def validate_market(self, markets):
        self.__market_closureness(markets)
