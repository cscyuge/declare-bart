from model import build
import torch
model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(24, True)
CLS = config.tokenizer.cls_token
SEP = config.tokenizer.sep_token

save_file_best = torch.load('./cache/best_save.data')
model.load_state_dict(save_file_best['para'])

# src = 'NSTEMI/CAD-history of 3V-CABG with only RCA graft still patent'
src = 'Weaned off vent to CPAP and was extubated in the afternoon on 9-2 by the pulmonary team .'
# src = '#cirrhosis: patient with history of alcoholic vsnash cirrhosis complicated by esophagel , gastric , and rectal varices'
src_tokens = config.tokenizer.tokenize(src)
src_tokens = [CLS] + src_tokens + [SEP]
print(src_tokens)
src_ids = config.tokenizer.convert_tokens_to_ids(src_tokens)

outputs = model.generate(input_ids=torch.tensor([src_ids]).to(config.device), do_sample=False,
                         max_length=config.pad_size)
outputs = config.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(outputs)

'''
BART:
NSTEMI/CAD-history of 3V-CABG with only RCA graft still patent
NSTEMI/coronary artery disease-history of 3V-catheter graft with only right coronary artery graft still patent

#cirrhosis: patient with history of alcoholic vsnash cirrhosis complicated by esophagel , gastric , and rectal varices
#cirrhosis: patient with history of alcoholic vsnash cirrhosis complicated by esophagel, gastric, and rectal varices
'''