import os

# Runtime controls must be set before importing TensorFlow through the RL module.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("JET_EXPERIMENT_SEED", "0")

import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import reinforcementlearning as dl

from viz import setup_viz, clean_viz
from simulation import run

def main(epochs: int = 1, sprint: bool = False) -> int:
    dl.configure_experiment_seed(int(os.environ["JET_EXPERIMENT_SEED"]))
    if not sprint:
        setup_viz()
    try:
        run(epochs=epochs, sprint=sprint)
        return 0
    except KeyboardInterrupt:
        print("\n===| Program interrupted by user |===")
        return 130
    finally:
        if not sprint:
            clean_viz()

if __name__ == "__main__":
    training = input("Would you like to train the model? (y/n): ").strip().lower() == "y"
    epochAmount = 1
    dl.TRAINING = training
    if training:
        epochAmount = 10000
    sys.exit(main(epochs=epochAmount, sprint=training))
    