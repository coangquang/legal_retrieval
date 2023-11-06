import torch
from torch import nn
from transformers import AutoModel

class CrossEncoder(nn.Module):
    def __init__(self, 
                 encoder= None,
                 model_checkpoint="vinai/phobert-base-v2",
                 representation=0,
                 fixed=False,
                 dropout=0.1):
        super(CrossEncoder, self).__init__()
        if encoder:
            self.encoder=encoder
        else:
            self.encoder = AutoModel.from_pretrained(model_checkpoint)
        self.representation = representation
        self.fixed = fixed
        self.classifier = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(self.encoder.config.hidden_size,2))

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None):
        
        if self.fixed:
            with torch.no_grad():
                outputs = self.encoder(input_ids,
                                       attention_mask, 
                                       token_type_ids)
                
                sequence_output = outputs['last_hidden_state']
                sequence_output = sequence_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
                if self.representation > -2:
                    output = sequence_output[:, self.representation, :]
                elif self.representation == -10:
                    output = sequence_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                elif self.representation == -100:
                    output = outputs[1]
            
                logits = self.classifier(output)
                #probabilities = nn.functional.softmax(logits, dim=-1)
        else:
            outputs = self.encoder(input_ids,
                                   attention_mask,
                                   token_type_ids)
            
            sequence_output = outputs['last_hidden_state']
            sequence_output = sequence_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
            if self.representation > -2:
                output = sequence_output[:, self.representation, :]
            elif self.representation == -10:
                output = sequence_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif self.representation == -100:
                output = outputs[1]
            
            logits = self.classifier(output)
            #probabilities = nn.functional.softmax(logits, dim=-1)
            
        return logits#, probabilities