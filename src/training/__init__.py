"""Training module — lazy imports to avoid CUDA/bitsandbytes issues"""

from src.training.synthetic_dialogue_generator import (
    SyntheticDialogueGenerator,
    create_synthetic_dataset,
)

# Heavy imports (trl, torch) are lazy — only imported when actually used
def train_sft(**kwargs):
    from src.training.sft_trainer import train_sft as _train_sft
    return _train_sft(**kwargs)

def train_rl(**kwargs):
    from src.training.rl_trainer import train_rl as _train_rl
    return _train_rl(**kwargs)

def feature_prediction_reward(*args, **kwargs):
    from src.training.rl_trainer import feature_prediction_reward as _fpr
    return _fpr(*args, **kwargs)
