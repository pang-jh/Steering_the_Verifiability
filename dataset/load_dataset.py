import os
import json

def load_dataset_single(file_path):

    with open(file_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    dataset = []
    for d in raw_data:
        text = d.get('text','').strip()
        image = d.get('image',None)
        gt = d.get('gt',None)
        pred = d.get('pred',None)
        dataset.append({
            'text':text,
            'image':image,
            'gt':gt,
            'pred':pred
        })
    return dataset

def load_dataset_split(file_path):

    with open(file_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    dataset = []
    for d in raw_data:
        image = d.get('image')
        text = d.get('text').strip()
        pred = d.get('pred')
        gt = d.get('gt')
        dataset.append({
            'image':image,
            'text':text,
            'pred':pred,
            'gt':gt
        })
    return dataset

def load_dataset_test(file_path):

    with open(file_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    dataset = []
    for d in raw_data:
        image = d.get('image')
        text = d.get('text').strip()
        pred = d.get('pred')
        gt = d.get('gt')
        dataset.append({
            'image':image,
            'text':text,
            'pred':pred,
            'gt':gt
        })
    return dataset
