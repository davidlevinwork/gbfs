from kneed import KneeLocator


class KneesLocator:
    """
    The KneesLocator class utilizes the KneeLocator from the kneed package to identify the knee point(s) in a curve.

    :param x: A list of x values of the curve.
    :param y: A list of y values of the curve.
    """

    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y

    def run(self) -> KneeLocator:
        """
        Executes the knee locating process using the provided x and y values.

        :return: A KneeLocator instance which can be used to access the knee value, knee y value,
                 and other information about the located knee point.
        """
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
