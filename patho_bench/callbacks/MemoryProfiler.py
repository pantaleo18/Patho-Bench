from torch.profiler import profile, ProfilerActivity, record_function
from typing import List
from pathlib import Path

class MemoryProfileAndSnapshot():

    _call_counter : int = 0

    def on_train_batch_start(self): pass

    def on_train_batch_end(self): pass