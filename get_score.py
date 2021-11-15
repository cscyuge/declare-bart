import pickle
from nlgeval import compute_metrics
import nltk
from utils.bleu_eval import count_hit,count_common

with open('./result/results.pkl','rb') as f:
    results = pickle.load(f)
with open('./data/test_dic.pkl','rb') as f:
    test_dics = pickle.load(f)
with open('./data/test/tar_txts.pkl','rb') as f:
    tar_txts = pickle.load(f)

tar_txts = tar_txts[len(tar_txts) // 2:]
results = results[len(results)//2:]

open('./tmp/hyp.txt', 'w', encoding='utf-8').writelines([x+ '\n' for x in results])
ref0 = [x[0] for x in tar_txts]
ref1 = [x[1] for x in tar_txts]
ref2 = [x[2] for x in tar_txts]
ref3 = [x[3] for x in tar_txts]
open('./tmp/ref0.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref0])
open('./tmp/ref1.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref1])
open('./tmp/ref2.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref2])
open('./tmp/ref3.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref3])
hyp = [nltk.word_tokenize(x) for x in results]

hit = count_hit(hyp, test_dics)
com = count_common(hyp)

metrics_dict=compute_metrics(hypothesis='tmp/hyp.txt',
                             references=['tmp/ref0.txt','tmp/ref1.txt','tmp/ref2.txt','tmp/ref3.txt'],
                             no_glove=True, no_overlap=False, no_skipthoughts=True)

BLEU = (metrics_dict['Bleu_1']+metrics_dict['Bleu_2']+metrics_dict['Bleu_3']+metrics_dict['Bleu_4'])/4
Ascore = (1+2.25+4)/(4/BLEU+2.25/hit+1/com)
print("BLEU:%4f, HIT:%4f, COM:%4f, ASCORE:%4f"%(BLEU, hit, com, Ascore))
