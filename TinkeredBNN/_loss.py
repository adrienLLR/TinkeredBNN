class BaseLoss:
    def J(self, a, y):
        return a - y
    
    def dJ(self, a, y):
        return 1

class EuclideanLoss:
    def J(self, o, y):
        return (((o-y)).dot((o-y).T))**0.5

    def dJ(self, o, y):
        return o-y