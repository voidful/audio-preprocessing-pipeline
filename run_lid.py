import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm

from lid_enhancement import AudioLIDEnhancer

lang2code = {
    'chinese': 'zh',
    'english': 'en',
    'french': 'fr',
    'german': 'de',
    'spanish': 'es'
}
code2label = {
    'zh': 1,
    'en': 2,
    'fr': 3,
    'de': 4,
    'es': 5
}
lang2label = {
    'chinese': 1,
    'english': 2,
    'french': 3,
    'german': 4,
    'spanish': 5
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default="/home/itk0123/crnn-lid/data/raw", help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=30, help="Number of workers")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    result_jsons = []
    for i in tqdm(nlp2.get_files_from_dir(source_dir, match='ogg')):
        try:
            result_jsons.append(i)
        except:
            pass
    
    wrong = []
    preds = []
    labels = []
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=True)
    for file_dir in tqdm(result_jsons, desc="LID"):
        rel_dir = os.path.relpath(file_dir, config['src'])
        label = lang2label[rel_dir.split('/')[0]]
        label_code = lang2code[rel_dir.split('/')[0]]
        result_code = lid_model(file_dir, possible_langs=['zh' ,'en', 'fr', 'de', 'es'])[0]
        result = code2label.get(result_code)
        if result == None:
            result = 0
        if result != label:
            wrong.append((result_code, label_code))
        preds.append(result)
        labels.append(label)

    print('Wrong predictions: ', [f'pred: {w[0]}, gt: {w[1]}' for w in wrong])
    print('Precision: ', precision_score(labels, preds, average=None))
    print('Recall: ', recall_score(labels, preds, average=None))
    print('Accuracy: ', accuracy_score(labels, preds))

