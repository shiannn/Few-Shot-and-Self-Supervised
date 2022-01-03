import torch
import torch.nn as nn
from torch.nn import functional as F
from config_p1 import CLASSES_PER_EPISODE, SAMPLES_PER_CLASS, N_SUPPORT, DEVICE

def prototypical_loss(model_output, target, metric='euc', relationNet=None):
    ### model_output CLASSES_PER_EPISODE* SAMPLES_PER_CLASS* dim
    model_output = model_output.reshape(CLASSES_PER_EPISODE, SAMPLES_PER_CLASS, -1)
    target = target.reshape(CLASSES_PER_EPISODE, SAMPLES_PER_CLASS)
    support_samples = model_output[:,:N_SUPPORT,:]
    #support_targets = target[:,:N_SUPPORT]
    query_samples = model_output[:,N_SUPPORT:,:]
    #query_targets = target[:,N_SUPPORT:]
    ### 000 111 222 333 444 555
    query_targets = torch.arange(CLASSES_PER_EPISODE).repeat_interleave(SAMPLES_PER_CLASS-N_SUPPORT).to(DEVICE)
    
    prototype = support_samples.mean(dim=1)
    ### prototype: CLASSES_PER_EPISODE* dim
    query_samples = query_samples.reshape(
        CLASSES_PER_EPISODE*(SAMPLES_PER_CLASS-N_SUPPORT),-1
    )
    #print(prototype.shape)
    #print(query_samples.shape)
    if metric == 'euc':
        pairdist=torch.cdist(query_samples, prototype, p=2)
    elif metric == 'cos':
        unit_query_samples = torch.nn.functional.normalize(query_samples, p=2)
        unit_prototype = torch.nn.functional.normalize(prototype, p=2)
        #print(unit_query_samples.shape)
        #print(unit_prototype.shape)
        cos_sim = torch.mm(unit_query_samples, unit_prototype.transpose(0, 1))
        pairdist = 1-cos_sim
    elif metric == 'parametric':
        if relationNet is None:
            print('need relation net')
            exit(0)
        repeat_query_samples = query_samples.unsqueeze(1).repeat(1,prototype.shape[0],1)
        repeat_prototypes = prototype.unsqueeze(0).repeat(query_samples.shape[0],1,1)
        #print(repeat_query_samples.shape)
        #print(repeat_prototypes.shape)
        relation_matrix = torch.cat([repeat_query_samples, repeat_prototypes], dim=2)
        #relation_matrix = (repeat_query_samples-repeat_prototypes)**2
        relation_matrix = relation_matrix.unsqueeze(0).permute(0,3,1,2)
        #print(relation_matrix.shape)
        #print(relationNet(relation_matrix).shape)
        pairdist = relationNet(relation_matrix)

    #print(pairdist)
    logits = F.log_softmax(-pairdist, dim=1)
    #print(logits)
    #print(query_targets)
    loss_val = -torch.gather(logits, dim=1, index=query_targets.unsqueeze(1))
    loss_val = loss_val.mean()
    #print(loss_val)
    _, max_idx = logits.max(dim=1)
    #print(max_idx)
    correct_num = (max_idx == query_targets).sum()
    
    return loss_val, correct_num