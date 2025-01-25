from transformers import AutoTokenizer, AutoConfig
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train import BertDataset
from eval import evaluate
from model import PLM_MTC
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--data', type=str, default='wos',  help='Dataset.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_micro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
args = parser.parse_args()


#def draw_tsne(text,label,truth):
#    mask = (labels == 1)

if __name__ == '__main__':

    ckp_path = os.path.join('../HyILR/data', args.data,'Checkpoints')
    data_path = os.path.join('../HyILR/data', args.data)

    checkpoint = torch.load(os.path.join(ckp_path, args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')       
    batch_size = args.batch
    device = args.device
    extra = args.extra
    mod_name=args.name
    args = checkpoint['args'] if checkpoint['args'] is not None else args


    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    config = AutoConfig.from_pretrained(args.backbone)

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    hier = torch.load(os.path.join(data_path, 'slot.pt'))
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)

    
    model = PLM_MTC(config,num_labels=num_class,backbone=args.backbone,bce_wt=args.bce_wt,curv_init=args.curv_init,
                    learn_curv=args.learn_curv,cl_loss=args.cl_loss,cl_wt=args.cl_wt,cl_temp=args.cl_temp)


    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model.load_state_dict(checkpoint['param'])

    model.to(device)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    text_hyp=[]

    model.eval()
    pbar = tqdm(test)
    with torch.no_grad():
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label )
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)
            
            for l in output['logits']:
                pred.append(torch.sigmoid(l).tolist())


    pbar.close()
    scores = evaluate(pred, truth, label_dict,)
    pred_rcv=np.array(pred)
    np_name=mod_name+extra+'.npy'
    #np.save(np_name,pred_rcv)

    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print(f'Model {mod_name} with best {extra} checkpoint')
    print('macro', macro_f1, 'micro', micro_f1)
    
