
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    filter_train: bool = True
    filter_val: bool = True
    max_new_tokens: int = 512

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)