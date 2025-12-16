class BruteForce:
    def __init__(self, dataset: list):
        self.dataset = dataset

    def L2Sqr(self, p: list, q: list) -> float:
        total = 0.0
        for xy in range(len(p)):
            total += (p[xy] - q[xy]) ** 2
        return total

    def RunSearch(self, queryset: list, topk: int) -> list[tuple[float, int]]:
        res = []
        for q in queryset:
            q_res = []
            for idx_d, d in enumerate(self.dataset):
                q_res.append((self.L2Sqr(d, q), idx_d))
            q_res.sort(key=lambda r: r[0])
            res.append(q_res[:topk])
        return res
