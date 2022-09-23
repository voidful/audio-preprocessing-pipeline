import argparse
import os

import ffmpeg
import nlp2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # thread_map


def audio_file_preprocessing_op(file_path):
    file_dir, filename_ext = os.path.split(file_path)
    filename = filename_ext.split('.')[0]
    if nlp2.is_file_exist(f'{file_dir}/{filename}.webm'):
        (
            ffmpeg
                .input(f'{file_dir}/{filename}.webm')
                .output(f'{file_dir}/{filename}.ogg', acodec='libvorbis', ar='16k', vn=None)
                .run(overwrite_output=1)
        )
        os.remove(f'{file_dir}/{filename}.webm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default="data", help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=30, help="Number of workers")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    result_jsons = []
    for i in tqdm(nlp2.get_files_from_dir(source_dir, match='webm')):
        try:
            result_jsons.append(i)
        except:
            pass

    process_map(audio_file_preprocessing_op, result_jsons, max_workers=config['workers'])
