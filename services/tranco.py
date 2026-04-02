import os

tranco_path = "data/tranco_top_1m.csv"
max_rank = 1_000_000

class TrancoService:
    def __init__(self, path=tranco_path):
        self.domain_to_rank = {}
        self._load(path)

    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tranco list not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rank, domain = line.split(",", 1)
                self.domain_to_rank[domain.lower()] = int(rank)

    def _normalize_rank(self, rank: int) -> float:
        """
        Normalize Tranco rank to [0, 1]
        1.0 = rank 1 (most trusted)
        0.0 = rank 1,000,000 or worse
        """
        return max(
            0.0,
            min(
                1.0,
                1.0 - (rank - 1) / (max_rank - 1)),
        )

    def lookup(self, domain: str) -> dict:
        #Returns Tranco signal for a domain
        domain = domain.lower().strip()

        if domain in self.domain_to_rank:
            rank = self.domain_to_rank[domain]
            score = self._normalize_rank(rank)

            return {
                "in_tranco": 1,
                "tranco_rank": rank,
                "tranco_score": round(score, 4),
            }

        return {
            "in_tranco": 0,
            "tranco_rank": None,
            "tranco_score": 0.0,
        }


# with open('data/tranco_top_1m.csv') as f:
#     TRANCO_SET = set(line.strip() for line in f)

# def get_tranco_score(domain: str) -> float:
#     # simple scoring: 1 if in top 1m, 0 otherwise
#     return 1.0 if domain in TRANCO_SET else 0.0
