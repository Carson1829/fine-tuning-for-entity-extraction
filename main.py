import json
import pandas as pd
from collections import defaultdict

from model import train_model
from inference import get_predictions, run_inference
from evaluate import get_f1_scores



def main():
    '''
    Uncomment the line to run
    '''
    # Part 1: few-shot prompting and F1 evaluation on validation set
    # get_predictions("val.txt", "few_shot_val_predictions.json", few_shot=True)
    # get_f1_scores("few_shot_val_predictions.json")

    # Get test predictions
    # get_predictions("test.txt", "few_shot_test_predictions.json", few_shot=True)


    # part 2:
    # Fine-tuning model 
    # train_model()
    
    # F1 evaluation on validation set
    # get_predictions("val.txt", "finetune_val_predictions.json", few_shot=False)
    # get_f1_scores("finetune_val_predictions.json")

    # Get test predictions
    # get_predictions("test.txt", "finetune_test_predictions.json", few_shot=False)

    # Get predictions on a single unannotated MMD file
    # run_inference("(mmd) Algebra - Lang.mmd.filtered", output_path="finetune_predictions_Lang.json")



if __name__ == "__main__":
    main()
