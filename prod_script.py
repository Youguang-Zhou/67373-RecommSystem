# 命令行调用示例：
# python prod_script.py '《黑暗之魂3》艾利德修女boss战——来自高端玩家的教学视频' '视频-游戏视频' 'abd8ef7f6c25447ba3ea164c627a105f' 12

import os
import pickle
import sys
from pathlib import Path

path = f'{Path(os.getcwd()).parent}/67373-RecommSystem'

NUM_RECOMM_VIDEOS = 12

try:
    input_title = sys.argv[1].strip()
    input_cate_name = sys.argv[2].strip()
    input_video_id = sys.argv[3].strip()
    NUM_RECOMM_VIDEOS = int(sys.argv[4].strip())
except:
    exit()


def load_data():
    data = []
    with open(f'{path}/data/data.txt') as f:
        for line in f.read().splitlines():
            [title, cate_name, cate_id, duration, creation_time, video_id] = line.split(',')
            data.append({
                'title': title,
                'cate_name': cate_name[3:],
                'cate_id': cate_id,
                'duration': duration,
                'creation_time': creation_time,
                'video_id': video_id,
            })
    return data


def load_model(fname):
    with open(f'{path}/models/{fname}', 'rb') as f:
        return pickle.load(f)


def tokenizer(s):
    return list(s)


data = load_data()

vectorizer = load_model('vectorizer.pkl')
knn = load_model('knn.pkl')


query = vectorizer.transform([''.join([input_title, input_cate_name])])
nbrs = knn.kneighbors(query, return_distance=False)
recomms = [(data[idx]['title'], data[idx]['video_id']) for idx in nbrs[0]]
recomms = filter(lambda v: v[1] != input_video_id, recomms)
recomms = list(map(lambda v: v[1], recomms))
if len(recomms) > NUM_RECOMM_VIDEOS:
    recomms = recomms[:NUM_RECOMM_VIDEOS]

print(recomms)
