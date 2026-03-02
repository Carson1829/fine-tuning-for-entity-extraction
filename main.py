import json
import pandas as pd
from collections import defaultdict

import config
from data import load_and_group, get_examples
from model import get_tokenizer, load_base_model, load_lora_model, train_model
from inference import get_predictions, run_inference
from evaluate import get_f1_scores
from few_shot import few_shot_predict_file


def main():

    # Part 1: few-shot prompting and F1 evaluation on validation set
    # run_few_shot(output_path="few_shot_val_predictions.json")
    # get_f1_scores("few_shot_val_predictions.json")

    # part 2:
    # Fine-tuning model 
    # train_model()
    # Get predictions on a single unannotated MMD file
    # run_inference("(mmd) Algebra - Lang.mmd.filtered", output_path="predictions_Lang.json")

    # F1 evaluation on validation set
    # get_val_predictions(output_path="val_predictions.json")
    # get_f1_scores("val_predictions.json")

    # Get predictions on the test set
    # get_test_predictions(output_path="test_predictions.json")

if __name__ == "__main__":
    main()
