from indicator import UnifiedQA_Scorer
from datasets import load_dataset
from tqdm import tqdm

def KDA_passage_score():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #I don't know if it works or not but i just want to set the device
    dataset = load_dataset("voidful/set-dg") 
    data_names=['eqg_race_f_dev','eqg_race_f_test']
    models=["allenai/unifiedqa-v2-t5-small-1251000","allenai/unifiedqa-v2-t5-base-1251000"]
    
    acc_score={}
    for model in models: #try small and base model to ompute the QA　score
        model_lib={}
        uni_score = UnifiedQA_Scorer(model_config=model) 
        for data_name in data_names: #compute score on different data
            data_lib={}
            for have_passage in range(2): #try KDA by have passage or not
                correct=0
                count=0
                err_count=0
                for a_data in tqdm(dataset[data_name]): #test each question
                    count+=1
                    qainput=''
                    if have_passage == 0: #there is no passage for question
                        qainput=a_data['question']
                    else:                   # there is a passage for question
                        qainput=a_data['passage']+'\\n'+a_data['question']
                    options=a_data['options']
                    answer=a_data['answer']
                    output=uni_score.option_score(qainput,options)  #output the score
                    output=list(output)
                    output_pos=output.index(max(output)) #check where the predicted answer is
                    if options[output_pos] == answer:
                        correct+=1
                    else:
                        err_count+=1
                print()
                print(f'model name = {model}, dataset name = {data_name}, hava passage = {have_passage}')
                print('總共 = ',count)
                print('正確 = ',correct)
                print('錯誤 = ',err_count)
                
                if have_passage == 0:       #store the score of passage 
                    data_lib['no_passage']=correct/count
                else:
                    data_lib['passage']=correct/count
            model_lib[data_name]=data_lib   #store the score of a data
        acc_score[model]=model_lib          #st
    return acc_score

def del_distractor_score():                
    #刪掉一個distractor之後分數是多少
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #I don't know if it works or not but i just want to set the device
    import copy
    
    dataset = load_dataset("voidful/set-dg")
    data_names=['eqg_race_f_dev','eqg_race_f_test']
    models=["allenai/unifiedqa-v2-t5-small-1251000","allenai/unifiedqa-v2-t5-base-1251000"]
    acc_score={}
    
    for model in models:                #try small and base model to ompute the QA　score
        model_lib={}
        uni_score = UnifiedQA_Scorer(model_config=model)
        for data_name in data_names:    #compute score on different data
            #data_lib={}
            count=0
            correct=0
            err_count=0
            for a_data in tqdm(dataset[data_name]):     #test each question
                
                qainput=a_data['passage']+'\\n'+a_data['question']
                options=a_data['options']
                answer=a_data['answer']
                index=a_data['answer_index']
                
                #ori_score=uni_score.option_score(qainput,options)[index]

                dis_index=[x for x in range(len(options))]
                dis_index.remove(index)         #remove the answer index so that we don't test without answer

                cnt=len(dis_index)
                for test in dis_index:
                    count+=1
                    test_option=copy.deepcopy(options)
                    test_option.pop(test)       #delete one distractor
                    sco=uni_score.option_score(qainput,test_option)     #score without a distractor
                    ans_ids=0
                    for i in range(len(test_option)):       #find the answer's new index
                        if test_option[i]==answer:
                            ans_ids=i
                    sco=list(sco)
                    if sco.index(max(sco))==ans_ids:
                        correct+=1
                    else:
                        err_count+=1
            print()
            print(f'model name = {model}, data name = {data_name}')
            print('總共 = ',count)
            print('正確 = ',correct)
            print('錯誤 = ',err_count) 
            #scores.append(correct/count)  
            #data_lib[data_name]=correct/count
            model_lib[data_name]=correct/count  
        acc_score[model]=model_lib
    return acc_score
KDA_score=KDA_passage_score()
#del_dis_score=del_distractor_score()
output=[KDA_score]
print(KDA_score)
#print(del_dis_score)
import json
with open('./scores.json','w',) as f:
    a=json.dump(output,f,ensure_ascii=False,)
