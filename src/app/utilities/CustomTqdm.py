from tqdm import tqdm


class CustomTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.callback = kwargs.pop("callback", None)
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self.callback and self.total:
            progress_fraction = self.n / self.total
            self.callback(progress_fraction)
