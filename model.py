
from transformers import AutoConfig,AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from criterion import CLloss




class LMEnc(nn.Module):
    
    def __init__(self,config,backbone='bert-base-uncased',):
        super(LMEnc, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = nn.Linear(config.hidden_size, config.num_labels, bias=False)
    
    def forward(self,input_ids, attention_mask):
        """m: batch size, c: no of labels, s: token sequence length, h : hidden size"""

        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state']
        bert_output = self.dropout(bert_output)
        masks = torch.unsqueeze(attention_mask, 1)  # (m, 1, s)
        attention = self.attention(bert_output).transpose(1, 2).masked_fill(~masks, -np.inf)  # (m, c, s)
        attention = F.softmax(attention, -1)
        representation = attention @ bert_output   #  (m,c,h)

        return representation, self.attention.weight, bert_output




class PLM_MTC(nn.Module):
    def __init__(self,config,num_labels,backbone,bce_wt,curv_init,learn_curv,level_dict,hier,cl_loss,cl_wt,cl_temp):
    
        super(PLM_MTC, self).__init__()
        self.num_labels=num_labels
        config.num_labels=num_labels
        self.bce_wt=bce_wt
        self.textenc=LMEnc(config,backbone)
        self.classifier = nn.Linear(num_labels*config.hidden_size, num_labels)
        self.cl_loss=cl_loss
        if self.cl_loss:
            self.cl_wt=cl_wt # weight of contrastive loss
            self.cl=CLloss(level_dict=level_dict,hier=hier,cl_temp=cl_temp,curv_init=curv_init,learn_curv=learn_curv,embed_dim=config.hidden_size) # initiliazing contarstive loss module
            

        
        
    def forward(self, input_ids, attention_mask,labels):
        
        """m: batch_size, c: no of labels, h: hidden_size, s: token sequnece length"""
    
        output,label_emb,bert_output = self.textenc(input_ids, attention_mask) # output (m,c,h), label_emb: (c,h), bert_output : (m,s,h)

        logits=self.classifier(output.view(output.shape[0],-1))
        
        loss=0
        
        if self.training:
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)
                loss += loss_fct(logits.view(-1, self.num_labels), target)*(self.bce_wt)
            
            
            if self.cl_loss:
                # contrastive loss which we be computed in hyperbolic space
                cl_loss=self.cl(bert_output[:,0,:],label_emb,target)
                loss+=cl_loss*self.cl_wt

        

        return {
            'loss': loss,
            'logits': logits,
            #'attentions': attns,
            #'hidden_states': label_aware_embedding,
            #'contrast_logits': contrast_logits,
            }
        






