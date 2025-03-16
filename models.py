import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import BaseModelOutput

class TextGuidedHarmonizationModel(nn.Module):
    def __init__(self, bart, roberta_model_name="roberta-base", hidden_dim=512, device=torch.device('cpu')):
        super().__init__()

        self.device = device
        
        # Load frozen RoBERTa
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta.to(device)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Get BART
        self.bart = bart
        self.bart.to(device)
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False  # Freeze BART encoder
        
        # Attention-based fusion layer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.attn.to(device)
        
        # LSTM for sequential refinement
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lstm.to(device)
        
        # Projection layer for RoBERTa conditioning vector
        self.condition_proj = nn.Linear(768, hidden_dim)
        self.condition_proj.to(device)
    # end init
        
    def forward(self, melody_input_ids, melody_attention_mask, harmony_input_ids, texts, labels=None):
        
        # Encode text guidance using frozen RoBERTa
        roberta_inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():  # Freeze RoBERTa (unless fine-tuning)
            roberta_outputs = self.roberta(**roberta_inputs)
        c = roberta_outputs.last_hidden_state[:, 0, :]  # Take [CLS] token representation
        c = self.condition_proj(c).unsqueeze(1)  # Shape: (batch, 1, hidden_dim)
        
        # Encode melody using frozen BART encoder
        with torch.no_grad():
            encoder_outputs = self.bart.model.encoder(input_ids=melody_input_ids, attention_mask=melody_attention_mask, return_dict=True)
        h = encoder_outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_dim)
        
        # Expand conditioning vector and apply attention fusion
        c_expanded = c.expand(-1, h.size(1), -1)  # Match sequence length
        hat_h, _ = self.attn(h, c_expanded, h)  # Attend RoBERTa conditioning vector to melody encoder states
        
        # LSTM-based fusion to refine text influence sequentially
        hat_h, _ = self.lstm(hat_h)  # Shape remains (batch, seq_len, hidden_dim)

        # encoder_outputs = BaseModelOutput(last_hidden_state=hat_h)

        print('hat_h: ', hat_h.shape)
        print('labels: ', labels.shape)
        
        # Decode harmony using BART decoder with fused hidden states
        decoder_outputs = self.bart(
            encoder_outputs=(hat_h, ),
            # encoder_outputs=encoder_outputs,
            attention_mask=melody_attention_mask,
            # decoder_input_ids=labels,
            decoder_input_ids=harmony_input_ids,
            labels=labels,
            return_dict=True
        )
        
        return decoder_outputs.loss, decoder_outputs.logits
    # end forward
# end class TextGuidedHarmonizationModel