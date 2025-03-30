import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN
from mingpt.model import GPT

class MedusaGPT(nn.Module):
    
    def __init__(self, config):
        # GPT Model
        super().__init__()
        self.config = config
        self.GPT = GPT(config.gpt)
        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.config.gpt.n_embd, self.config.gpt.vocab_size) for _ in range(self.config.medusa.num_heads)
        ])

        # Initialise Medusa Heads with Weights of GPT's LM Layer
        for i in range(self.config.medusa.num_heads):
            self.medusa_heads[i].weight.data.copy_(self.GPT.lm_head.weight.data)
            # self.medusa_heads[i].bias.data.copy_(self.GPT.lm_head.bias.data)

    def forward(self, idx, gpt_target=None, medusa_targets=None):
        transformer_output, gpt_logits, gpt_loss = self.GPT(idx, gpt_target)
        
        medusa_logits = []
        for i in range(self.config.medusa.num_heads):
            medusa_logits.append(self.medusa_heads[i](transformer_output))
        
        medusa_logits = torch.stack(medusa_logits, dim=1)

        medusa_losses = torch.stack([
            self.config.medusa._lambdas[i] * F.cross_entropy(
                medusa_logits[i].view(-1, medusa_logits.size(-1)), 
                medusa_targets[i].view(-1), 
                ignore_index=-1
            )
            for i in range(self.config.medusa.num_heads)
        ])

        medusa_loss = medusa_losses.sum()

        return gpt_logits, medusa_logits, medusa_loss
    
    def configure_optimizers(self, config):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=config.train.learning_rate
        )
        return optimizer
    
    @staticmethod
    def get_default_config():
        C = CN()

        # system
        C.system = CN()
        C.system.seed = 3407
        C.system.work_dir = './out/chargpt'

        # model
        C.gpt = GPT.get_default_config()
        C.gpt.model_type = 'gpt-mini'
        
        # medusa
        C.medusa = CN()
        C.medusa.num_heads = 5 # Based on the paper.
        # Regularization parameters for loss of each head as mentioned in the OG paper.
        l = 0.8
        C.medusa._lambdas = [l, l**2, l**3, l**4, l**5]

        return C