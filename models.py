import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import BaseModelOutput
from copy import deepcopy

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
        
        return {
            'loss':decoder_outputs.loss,
            'logits': decoder_outputs.logits
        }
    # end forward

    def generate(self, merged_tokenizer, melody_input_ids, melody_attention_mask, texts, max_length, num_bars, temperature):
        batch_size = melody_input_ids.shape[0]
        bos_token_id = merged_tokenizer.bos_token_id
        eos_token_id = merged_tokenizer.config.eos_token_id
        bar_token_id = merged_tokenizer.vocab['<bar>']
        bars_left = deepcopy(num_bars)
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(self.device)  # (batch_size, 1)
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # (batch_size,)
        # Encode text guidance using frozen RoBERTa
        roberta_inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():  # Freeze RoBERTa (unless fine-tuning)
            roberta_outputs = self.roberta(**roberta_inputs)
            c = roberta_outputs.last_hidden_state[:, 0, :]  # Take [CLS] token representation
            c = self.condition_proj(c).unsqueeze(1)  # Shape: (batch, 1, hidden_dim)
            encoder_outputs = self.bart.model.encoder(input_ids=melody_input_ids, attention_mask=melody_attention_mask, return_dict=True)
            h = encoder_outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_dim)
            c_expanded = c.expand(-1, h.size(1), -1)
            hat_h, _ = self.attn(h, c_expanded, h)
            hat_h, _ = self.lstm(hat_h)
        for _ in range(max_length):
            # Pass through the decoder
            decoder_outputs = self.bart(
                encoder_outputs=(hat_h,),
                input_ids=decoder_input_ids,
                attention_mask=melody_attention_mask,
                return_dict=True
            )

            # Get the logits of the last generated token
            logits = decoder_outputs.logits[:, -1, :]  # Get next-token logits
            print('bars_left:', bars_left)
            # For the batch that has some bars left, zero out the eos_token_id logit
            # For the batch that has 0 bars left, zero out the bar token
            if bars_left != -1 and bar_token_id != -1:
                logits[ bars_left[:,0] > 0 , eos_token_id ] = 0
                logits[ bars_left[:,0] <= 0 , bar_token_id ] = 0

            # Apply temperature scaling and softmax
            probs = F.softmax(logits / temperature, dim=-1)  # (batch_size, vocab_size)

            # Sample next token
            next_token_ids = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            if bars_left != -1 and bar_token_id != -1:
                bars_left[ next_token_ids == bar_token_id ] -= 1

            # Stop condition: mask finished sequences
            finished |= next_token_ids.squeeze(1) == eos_token_id

            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)  # (batch_size, seq_len)

            # If all sequences are finished, stop early
            if finished.all():
                break
        return decoder_input_ids
    # end generate
# end class TextGuidedHarmonizationModel