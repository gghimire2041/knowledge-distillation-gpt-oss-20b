"""
Optimized student model architecture for Nepali language.
Designed for efficient inference while maintaining high quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Tuple, Union
import math

class NepaliStudentConfig(PretrainedConfig):
    """Configuration class for Nepali Student Model."""
    
    model_type = "nepali_student"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        dropout_rate=0.1,
        attention_dropout=0.1,
        activation_function="gelu",
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.initializer_range = initializer_range
        self.use_cache = use_cache


class NepaliMultiHeadAttention(nn.Module):
    """Optimized multi-head attention for Nepali processing."""
    
    def __init__(self, config: NepaliStudentConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-value for caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.output(context)
        
        return output, present_key_value


class NepaliMLP(nn.Module):
    """Feed-forward network optimized for Nepali language patterns."""
    
    def __init__(self, config: NepaliStudentConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Activation function
        if config.activation_function == "gelu":
            self.activation = F.gelu
        elif config.activation_function == "relu":
            self.activation = F.relu
        elif config.activation_function == "silu":
            self.activation = F.silu
        else:
            self.activation = F.gelu
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        return hidden_states


class NepaliTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config: NepaliStudentConfig):
        super().__init__()
        self.attention = NepaliMultiHeadAttention(config)
        self.mlp = NepaliMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attention_output, present_key_value = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        attention_output = self.dropout(attention_output)
        hidden_states = residual + attention_output
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.dropout(mlp_output)
        hidden_states = residual + mlp_output
        
        return hidden_states, present_key_value


class NepaliStudentModel(PreTrainedModel):
    """Lightweight student model specialized for Nepali language."""
    
    config_class = NepaliStudentConfig
    base_model_prefix = "nepali_student"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: NepaliStudentConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NepaliTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings to output weights for parameter efficiency
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if configured
        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _init_weights(self, module):
        """Initialize model weights using standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self):
        return self.token_embedding
    
    def set_input_embeddings(self, value):
        self.token_embedding = value
        self.lm_head.weight = value.weight  # Keep tied
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def _create_causal_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        batch_size, seq_len = input_ids.size()
        
        # Create position ids if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask
        if attention_mask is None:
            causal_mask = self._create_causal_mask(batch_size, seq_len, input_ids.device)
        else:
            # Combine with provided attention mask
            causal_mask = self._create_causal_mask(batch_size, seq_len, input_ids.device)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            causal_mask = causal_mask + attention_mask
        
        # Forward through transformer blocks
        present_key_values = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=False)
                    return custom_forward
                
                hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    causal_mask,
                    past_key_value,
                )
            else:
                hidden_states, present_key_value = block(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
        
        # Final layer norm and output projection
        hidden_states = self.ln_final(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (present_key_values,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    def generate_text(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the model with various decoding strategies."""
        
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Set default token IDs
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                if step == 0:
                    outputs = self.forward(input_ids=generated_ids, use_cache=True)
                    next_token_logits = outputs.logits[:, -1, :]
                else:
                    outputs = self.forward(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                
                past_key_values = outputs.past_key_values
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or select next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, 1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update finished sequences
                if eos_token_id is not None:
                    finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                
                # Stop if all sequences are finished
                if finished.all():
                    break
        
        return generated_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_footprint(self) -> dict:
        """Get model memory footprint information."""
        total_params = self.count_parameters()
        memory_mb = total_params * 4 / (1024 ** 2)  # Assume float32
        
        return {
            'total_parameters': total_params,
            'memory_mb': round(memory_mb, 2),
            'memory_gb': round(memory_mb / 1024, 3)
        }
