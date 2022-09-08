import datasets
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import jsonlines
from sklearn.metrics import confusion_matrix
NUM_PREPROCESSING_WORKERS = 2
# python3 run.py --do_train --output_dir ./trained_model/  --num_train_epochs 1  --per_device_train_batch_size 128
#--do_train --output_dir ./trained_model/ --num_train_epochs 10  --per_device_train_batch_size 48
# python3 run.py --do_eval --model ./trained_model/ --output_dir ./eval_output/
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Dataset selection
if "squad".endswith('.json') or "squad".endswith('.jsonl'):
    dataset_id = None
    # Load from local json/jsonl file
    dataset = datasets.load_dataset('json', data_files="squad")
    # By default, the "json" dataset loader places all examples in the train split,
    # so if we want to use a jsonl file for evaluation we need to get the "train" split
    # from the loaded datasetf
    eval_split = 'train'
else:
    default_datasets = {'qa': ('squad',), 'nli': ('squad',)}
    dataset_id = tuple("squad".split(':')) if "squad" is not None else \
        default_datasets["nli"]
    # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
    eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
    # Load the raw data
    dataset = datasets.load_dataset(*dataset_id)
# index=[0]*4
print(dataset['train'])
# print(index)

# average_permise_len=0
# average_hypothesis_len=0
# for i in  (dataset['train']):
#     average_permise_len+=len(i['premise'])
#     average_hypothesis_len+=len(i['hypothesis'])
# print(average_permise_len/dataset['train'].num_rows)
# print(average_hypothesis_len/dataset['train'].num_rows)
# for i in (dataset['train']['label']):

# NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")

label=[]
predict=[]
with jsonlines.open('./eval_output/eval_predictions.jsonl') as reader:
    for obj in reader:
        label_temp=obj['label']
        predict_temp=obj['predicted_label']
        if  label_temp!= predict_temp:
            print("-----------------------------------------------")
            print(obj['premise'])
            print(obj['hypothesis'])
            print(label_temp)
            print(predict_temp)
            print("-----------------------------------------------")
        label.append(label_temp)
        predict.append(predict_temp)
c_matrix=confusion_matrix(label, predict)
print(c_matrix)
