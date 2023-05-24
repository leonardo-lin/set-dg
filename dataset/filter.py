import json
import os

import nlp2

for jfile in nlp2.get_files_from_dir('./unfiltered/'):
    print(jfile)
    datas = []
    with open(jfile) as f:
        for line in f:
            data = json.loads(line)
            if '_' in data['question'] \
                    or all([(len(i.split(" ")) == 1) for i in data['options']]) \
                    or any([('none of the' in i.lower() or 'all of the' in i.lower()) and len(i.strip().split(" ")) < 7 for i in
                            data['options']]) \
                    or "NOT" in data['question']:
                print(data['question'], data['options'])
            else:
                datas.append(data)

    jfolder = os.path.dirname(jfile.replace('unfiltered', 'filtered'))
    nlp2.get_dir_with_notexist_create(jfolder)

    with open(jfile.replace('unfiltered', 'filtered'), 'w') as f:
        for data in datas:
            f.write(json.dumps(data) + '\n')

