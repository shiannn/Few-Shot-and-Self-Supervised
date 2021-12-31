import torch
from torch.nn import functional as F
from config_p1 import CLASSES_PER_EPISODE, SAMPLES_PER_CLASS, N_SUPPORT, DEVICE

def prototypical_loss(model_output, target):
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
    pairdist=torch.cdist(query_samples, prototype, p=2)
    #print(pairdist)
    logits = F.log_softmax(-pairdist, dim=1)
    #print(logits)
    #print(query_targets)
    loss_val = -torch.gather(logits, dim=1, index=query_targets.unsqueeze(1))
    loss_val = loss_val.mean()
    #print(loss_val)
    _, max_idx = logits.max(dim=1)
    #print(max_idx)
    acc_val = (max_idx == query_targets).float().mean()
    #print(acc_val)
    
    return loss_val, acc_val