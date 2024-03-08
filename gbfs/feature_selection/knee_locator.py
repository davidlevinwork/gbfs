from kneed import KneeLocator


class KneesLocator:
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y

    def run(self):
        knee_locator = KneeLocator(
            x=self.x,
            y=self.y,
            S=1.0,
            online=False,
            curve='concave',
            direction='increasing',
            interp_method='interp1d',
        )
        return knee_locator
