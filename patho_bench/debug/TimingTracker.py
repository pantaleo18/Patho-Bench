# utils/timing.py (or inline for now)
import time
from collections import defaultdict

class TimingTracker:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def tic(self, key):
        return time.perf_counter(), key

    def toc(self, tic_tuple):
        start, key = tic_tuple
        self.times[key] += time.perf_counter() - start
        self.counts[key] += 1

    def report(self, prefix=""):
        print(f"\n{prefix} Timing report:")
        for k in sorted(self.times):
            avg = self.times[k] / max(self.counts[k], 1)
            print(f"  {k:30s} | total {self.times[k]:.3f}s | avg {avg:.4f}s | n={self.counts[k]}")
