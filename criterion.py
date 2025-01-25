import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import os
import lorentz
import math


# code for exponential map projection and geodesic distance is in lorentz.py




class CLloss(torch.nn.Module):
    
    def __init__(self,level_dict,hier,cl_temp,curv_init=1,learn_curv=True,embed_dim=768):
        
        """Args:
        level_dict (dict): A dictionary where keys represent levels in the hierarchy,
            and values are lists of labels associated with each level.
            Example: 
                {1: [1, 2, 4], 2: [3, 4, 5]}
            Explanation:
                - Level 1 contains labels 1, 2, and 4.
                - Level 2 contains labels 3, 4, and 5.

        hier (dict): A dictionary representing parent-child relationships among labels.
            Example: 
                {1: {2, 4, 5}, 2: {8}}
            Explanation:
                - Label 1 is the parent of labels 2, 4, and 5.
                - Label 2 is the parent of label 8.

        cl_temp (float): Temperature parameter to scale similarity scores for contrastive loss.
        curv_init (float): Initial curvature of the hyperbolic space. Controls how "curved" the space is.
        learn_curv (bool): If True, allows the curvature to be optimized during training.
        embed_dim (int): Dimensionality of the input embeddings."""

        super(CLloss, self).__init__()
        self.level_dict=level_dict
        self.hier=hier
        self.cl_temp=cl_temp

        # Initialize curvature for lorentz model
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.text_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.label_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
    
    
    def collect_neg_siblings(self, pos_label, level_dict, pos_labels):
        """
        Get all sibling label nodes at the same level as the given label.
        
        Args:
            pos_label (int): The label for which to find siblings.
            level_dict (dict): Dictionary mapping levels to lists of labels.
            pos_labels (list): List of positive labels to exclude.

        Returns:
            list: A list of sibling labels that are not in pos_labels.
        """
        # A dictionary where each label (key) is mapped to its corresponding level (value) based on level_dict.
        label_level = {l: k for k, v in level_dict.items() for l in v}
        
        # Determine the level of the given label
        level = label_level[pos_label]
        
        # Get all labels at the same level
        all_labels_level = level_dict[level]
        
        # If only one label exists at this level, return no siblings
        if len(all_labels_level) == 1:
            return []
        
        # Exclude positive labels to get hard siblings
        hard_siblings = [l for l in all_labels_level if l not in pos_labels]
        return hard_siblings

    
    def collect_neg_desc(self, pos, hier, pos_labels):
        """
        Collects all descendant labels of a given label (pos) in the hierarchy,
        excluding labels already in pos_labels.

        Args:
            pos (int): The current label for which descendants are collected.
            hier (dict): Dictionary representing parent-child relationships.
            pos_labels (list): List of positive labels to exclude.

        Returns:
            set: A set of descendant labels for the given label, excluding pos_labels.
        """
        # Get direct descendants of the current label
        collected_desc = set(hier.get(pos, []))
        
        # Exclude positive labels
        collected_desc = {i for i in collected_desc if i not in pos_labels}

        # Recursively collect descendants of the current descendants
        for value in list(collected_desc):
            if value in hier:
                collected_desc.update(self.collect_neg_desc(value, hier, pos_labels))
        
        return collected_desc

    
    def forward(self, text_embeddings, label_embeddings, target_labels):

        """ text_embeddings: (m,h); label_embeddings: (c,h); target_labels: (m,c) multi-hot encoded"""
        # m: batch size, c: no of labels, h: hidden size

        _curv = self.curv.exp()

	    # multiply by scalars
        text_embeddings = text_embeddings * self.text_alpha.exp()
        label_embeddings = label_embeddings* self.label_alpha.exp()

        # Project into hyperbolic space
        text_hyplr=  lorentz.exp_map0(text_embeddings,self.curv.exp())
        label_hyplr= lorentz.exp_map0(label_embeddings,self.curv.exp())

       
        batch_size = text_embeddings.size(0)
        

        # Calculate geodesic distance between text  and label embeddings
    
        dist_sim_matrix=lorentz.pairwise_dist(text_hyplr,label_hyplr,_curv)



        # Identify positive labels for each text sample
        positive_labels = [torch.nonzero(label).view(-1).tolist() for label in target_labels]

        
        # Find hard negative labels
        
        hard_negative_labels_batch = []
        
        for i, pos_labels in enumerate(positive_labels):
            hard_negative_labels_sample = []
            negative_similarities = dist_sim_matrix[i].clone()
            
            
            for pos in pos_labels:
                # Find hard sibling negatives (labels at the same level but not positive)
                neg_siblings = self.collect_neg_siblings(pos,self.level_dict, pos_labels)
                
                if neg_siblings:
                    
                    # Get label with min geodesic distance  from negative siblings for i-th text sample

                    sib_idx = min(neg_siblings, key=lambda idx: negative_similarities[idx])
                    
                    # A negative descendant selected in a previous iteration might appear as a negative sibling 
                    # for another positive label at the next level. In such cases, the same label might be selected again.
                    # Handling this edge case
                    if sib_idx in hard_negative_labels_sample and len(neg_siblings)>1  and any(idx not in hard_negative_labels_sample for idx in neg_siblings):
                        # Remove the first minimum from consideration and find the next smallest sibling
                        sib_idx = min(
                            (idx for idx in neg_siblings if idx not in hard_negative_labels_sample),
                            key=lambda idx: negative_similarities[idx],
                        )

                    # Add the selected sibling to the hard negatives
                    hard_negative_labels_sample.append(sib_idx)

                    

                # Find hard descendant negatives (descendant labels excluding positives)
            
                neg_descendants = list(self.collect_neg_desc(pos, self.hier,pos_labels))

                if neg_descendants:
                    
                    # Get label with min geodesic distance  from negative descendants for i-th text sample
                    
                    desc_idx = min(neg_descendants, key=lambda idx: negative_similarities[idx])

                    # add it as hard negative for the sample
                    hard_negative_labels_sample.append(desc_idx)
                
            # remove duplicate neagtive labels if any 
            hard_negative_labels_sample=list(set(hard_negative_labels_sample))
            hard_negative_labels_batch.append(hard_negative_labels_sample)
        
      
        # Calculate contrastive loss
        loss = 0
        
        for i in range(batch_size):
            zi = text_hyplr[i]
            pos_indices, neg_index = positive_labels[i], hard_negative_labels_batch[i]

            # Calculate positive alignment scores
            pos_alignment_scores =-dist_sim_matrix[i, pos_indices]/self.cl_temp    

            
            # Calculate negative alignment score
            neg_alignment_scores = -dist_sim_matrix[i, neg_index]/self.cl_temp 

     
            denom= torch.cat([torch.exp(pos_alignment_scores), torch.exp(neg_alignment_scores)]).sum()
            pos_loss = -torch.log(torch.exp(pos_alignment_scores) /denom) 
            pos_loss=pos_loss.mean()
            loss += pos_loss




        # Average loss over the batch
        loss /= batch_size

        return loss


