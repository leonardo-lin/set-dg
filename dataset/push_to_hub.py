import os

import datasets
import nlp2
from datasets import Dataset

d_dict = {}
for jfile in nlp2.get_files_from_dir('./filtered/'):
    jfolder = os.path.dirname(jfile).split('/')[-1]
    jname = os.path.basename(jfile).split('.')[0]
    print(jfile, jfolder, jname)
    d_dict[jfolder + '_' + jname] = Dataset.from_json(jfile)

dd = datasets.DatasetDict(d_dict)
dd.push_to_hub('voidful/set-dg')
