"""
Custom Nepali tokenizer using SentencePiece, optimized for Devanagari script.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import sentencepiece as spm

logger = logging.getLogger(__name__)


class NepaliTokenizer:
    """Custom tokenizer optimized for Nepali language and Devanagari script."""
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 32000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None
        
        # Special tokens optimized for Nepali
        self.special_tokens = {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'mask_token': '<mask>',
            'cls_token': '<cls>',
            'sep_token': '<sep>'
        }
        
        # Nepali-specific tokens
        self.nepali_special_tokens = {
            'danda': '।',  # Nepali sentence delimiter
            'double_danda': '॥',  # Nepali paragraph delimiter
            'om': 'ॐ',  # Sacred symbol
            'rupee': '₹',  # Currency symbol
        }
        
        # Token IDs (will be set after training/loading)
        self.token_to_id = {}
        self.id_to_token = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_tokenizer(self, texts: List[str], output_dir: str, vocab_size: Optional[int] = None) -> str:
        """Train SentencePiece tokenizer on Nepali data."""
        logger.info("🔤 Training Nepali tokenizer...")
        
        if vocab_size is not None:
            self.vocab_size = vocab_size
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training data
        temp_file = output_dir / "temp_training_data.txt"
        logger.info(f"📝 Writing {len(texts)} texts to temporary file...")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                if text.strip():  # Only write non-empty texts
                    f.write(text.strip() + '\n')
        
        # Model output path
        model_prefix = output_dir / "nepali_tokenizer"
        
        # Prepare all special tokens
        all_special_tokens = list(self.special_tokens.values()) + list(self.nepali_special_tokens.values())
        
        # SentencePiece training arguments optimized for Nepali
        training_args = {
            'input': str(temp_file),
            'model_prefix': str(model_prefix),
            'vocab_size': self.vocab_size,
            'character_coverage': 0.99995,  # High coverage for Devanagari
            'model_type': 'bpe',  # Byte-pair encoding
            'max_sentence_length': 4192,
            'shuffle_input_sentence': True,
            'split_by_unicode_script': True,  # Important for mixed scripts
            'split_by_number': True,
            'split_by_whitespace': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'split_digits': True,
            'normalization_rule_name': 'nfkc',  # Unicode normalization
            'user_defined_symbols': all_special_tokens,
            'hard_vocab_limit': False,
            'use_all_vocab': True,
            'byte_fallback': True,  # Handle unknown characters gracefully
            'vocabulary_output_piece_score': True,
            'train_extremely_large_corpus': False,
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'num_threads': os.cpu_count() or 4,
        }
        
        try:
            spm.SentencePieceTrainer.train(**training_args)
            logger.info("✅ SentencePiece training completed")
        except Exception as e:
            logger.error(f"❌ SentencePiece training failed: {e}")
            raise
        
        # Load the trained model
        model_path = str(model_prefix) + ".model"
        self.load_model(model_path)
        
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()
        
        # Save tokenizer configuration
        self._save_config(output_dir)
        
        logger.info(f"✅ Nepali tokenizer trained and saved to: {model_prefix}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load trained SentencePiece model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        
        self.sp = smp.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path
        
        # Build token mappings
        self._build_token_mappings()
        
        logger.info(f"✅ Loaded tokenizer from: {model_path}")
        logger.info(f"📊 Vocabulary size: {self.sp.vocab_size()}")
    
    def _build_token_mappings(self):
        """Build token to ID and ID to token mappings."""
        if self.sp is None:
            return
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        for i in range(self.sp.vocab_size()):
            token = self.sp.id_to_piece(i)
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def encode(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], Dict[str, torch.Tensor]]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        # Add BOS token if requested
        if add_special_tokens:
            text = self.special_tokens['bos_token'] + text
        
        # Encode using SentencePiece
        token_ids = self.sp.encode(text, out_type=int)
        
        # Add EOS token if requested
        if add_special_tokens:
            eos_id = self.sp.piece_to_id(self.special_tokens['eos_token'])
            if eos_id != self.sp.unk_id():  # Only add if EOS token exists
                token_ids.append(eos_id)
        
        # Truncate if max_length specified
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # Ensure EOS token at the end if truncated
            if add_special_tokens:
                eos_id = self.sp.piece_to_id(self.special_tokens['eos_token'])
                if eos_id != self.sp.unk_id():
                    token_ids[-1] = eos_id
        
        # Return as tensors if requested
        if return_tensors == "pt":
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.ones(len(token_ids), dtype=torch.long)
            }
        elif return_tensors is not None:
            raise ValueError(f"Unsupported return_tensors format: {return_tensors}")
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Remove special tokens if requested
        if skip_special_tokens:
            special_token_ids = set()
            for token in self.special_tokens.values():
                token_id = self.sp.piece_to_id(token)
                if token_id != self.sp.unk_id():
                    special_token_ids.add(token_id)
            
            # Also remove PAD tokens
            pad_id = self.sp.piece_to_id(self.special_tokens['pad_token'])
            if pad_id != self.sp.unk_id():
                special_token_ids.add(pad_id)
            
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        # Decode using SentencePiece
        decoded_text = self.sp.decode(token_ids)
        
        return decoded_text.strip()
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts."""
        
        # Encode each text
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            
            # Truncate if needed
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # Ensure EOS token at the end
                if add_special_tokens:
                    eos_id = self.sp.piece_to_id(self.special_tokens['eos_token'])
                    if eos_id != self.sp.unk_id():
                        token_ids[-1] = eos_id
            
            all_token_ids.append(token_ids)
        
        # Pad sequences if requested
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_token_ids)
            
            pad_token_id = self.sp.piece_to_id(self.special_tokens['pad_token'])
            if pad_token_id == self.sp.unk_id():
                pad_token_id = 0  # Fallback to 0 if PAD token not found
            
            # Pad all sequences
            padded_token_ids = []
            attention_masks = []
            
            for token_ids in all_token_ids:
                attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
                token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
                
                padded_token_ids.append(token_ids)
                attention_masks.append(attention_mask)
            
            all_token_ids = padded_token_ids
        else:
            # Create attention masks without padding
            attention_masks = [[1] * len(ids) for ids in all_token_ids]
        
        # Convert to tensors
        result = {
            'input_ids': torch.tensor(all_token_ids, dtype=torch.long),
        }
        
        if padding or len(set(len(mask) for mask in attention_masks)) == 1:
            result['attention_mask'] = torch.tensor(attention_masks, dtype=torch.long)
        
        return result
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Batch decode multiple sequences."""
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return [self.decode(seq, skip_special_tokens) for seq in sequences]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp is None:
            return self.vocab_size
        return self.sp.vocab_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as token to ID mapping."""
        if self.sp is None:
            return {}
        return self.token_to_id.copy()
    
    def save_pretrained(self, output_dir: str):
        """Save tokenizer to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_path and Path(self.model_path).exists():
            # Copy model file
            import shutil
            target_path = output_dir / "tokenizer.model"
            shutil.copy2(self.model_path, target_path)
            logger.info(f"💾 Copied tokenizer model to: {target_path}")
        
        # Save configuration
        self._save_config(output_dir)
        
        logger.info(f"✅ Tokenizer saved to: {output_dir}")
    
    def _save_config(self, output_dir: Path):
        """Save tokenizer configuration."""
        config = {
            'tokenizer_type': 'NepaliTokenizer',
            'vocab_size': self.vocab_size,
            'model_type': 'sentencepiece',
            'special_tokens': self.special_tokens,
            'nepali_special_tokens': self.nepali_special_tokens,
        }
        
        config_path = output_dir / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save special tokens map
        special_tokens_map = {}
        if self.sp:
            for name, token in self.special_tokens.items():
                token_id = self.sp.piece_to_id(token)
                if token_id != self.sp.unk_id():
                    special_tokens_map[name] = {
                        'content': token,
                        'id': token_id,
                        'single_word': False,
                        'lstrip': False,
                        'rstrip': False,
                        'normalized': True
                    }
        
        if special_tokens_map:
            tokens_path = output_dir / "special_tokens_map.json"
            with open(tokens_path, 'w', encoding='utf-8') as f:
                json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'NepaliTokenizer':
        """Load tokenizer from pretrained directory."""
        model_path = Path(model_path)
        
        # Load configuration
        config_path = model_path / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Find model file
        tokenizer_model = model_path / "tokenizer.model"
        if not tokenizer_model.exists():
            # Try alternative names
            for alt_name in ["sentencepiece.model", "tokenizer.spm"]:
                alt_path = model_path / alt_name
                if alt_path.exists():
                    tokenizer_model = alt_path
                    break
        
        if not tokenizer_model.exists():
            raise FileNotFoundError(f"No tokenizer model found in {model_path}")
        
        # Create and load tokenizer
        tokenizer = cls(
            model_path=str(tokenizer_model),
            vocab_size=config.get('vocab_size', 32000)
        )
        
        # Update special tokens from config
        if
