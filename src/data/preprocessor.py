"""
Data preprocessing pipeline for Nepali text data.
Handles cleaning, chunking, and quality filtering.
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict
from langdetect import detect, LangDetectException
import logging

logger = logging.getLogger(__name__)


class NepaliPreprocessor:
    """Comprehensive preprocessor for Nepali text data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_length = config.get('max_length', 1024)
        self.min_length = config.get('min_length', 50)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Nepali text patterns
        self.devanagari_range = ('\u0900', '\u097F')  # Devanagari Unicode block
        self.nepali_punctuation = ['।', '॥', '?', '!', '.', ',', ';', ':']
        
        # Quality thresholds
        self.min_nepali_ratio = 0.6  # Minimum ratio of Nepali characters
        self.max_repeat_ratio = 0.1  # Maximum ratio of repeated characters
        self.min_sentence_count = 2   # Minimum sentences per text
        
        # Cleaning patterns
        self.setup_cleaning_patterns()
        
        logger.info(f"Initialized preprocessor: max_length={self.max_length}, min_length={self.min_length}")
    
    def setup_cleaning_patterns(self):
        """Setup regex patterns for text cleaning."""
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number pattern (Nepali format)
        self.phone_pattern = re.compile(
            r'(\+977[-.\s]?)?(98\d{8}|97\d{8}|96\d{8}|01[-.\s]?\d{7})'
        )
        
        # Excessive whitespace
        self.whitespace_pattern = re.compile(r'\s{3,}')
        
        # Repeated punctuation
        self.repeat_punct_pattern = re.compile(r'([।॥!?.,;:])\1{2,}')
        
        # HTML tags (if any remain from web scraping)
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Social media handles
        self.social_handle_pattern = re.compile(r'@\w+|#\w+')
        
        # Numbers (for optional filtering)
        self.number_pattern = re.compile(r'\d+')
    
    def process(self, raw_data: Dict[str, List[str]]) -> DatasetDict:
        """Process all raw data into a clean dataset."""
        
        logger.info("🧹 Starting comprehensive data preprocessing...")
        
        all_texts = []
        processing_stats = {
            'total_input': 0,
            'after_cleaning': 0,
            'after_quality_filter': 0,
            'after_chunking': 0,
            'final_count': 0,
            'sources': {}
        }
        
        # Process each source
        for source, texts in raw_data.items():
            logger.info(f"📝 Processing {len(texts)} texts from {source}...")
            
            source_stats = {
                'input_count': len(texts),
                'cleaned_count': 0,
                'quality_passed': 0,
                'chunks_created': 0
            }
            
            processing_stats['total_input'] += len(texts)
            
            processed_texts = []
            
            for text in texts:
                # Clean the text
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    source_stats['cleaned_count'] += 1
                    
                    # Apply quality filters
                    if self.passes_quality_check(cleaned_text):
                        source_stats['quality_passed'] += 1
                        
                        # Create chunks if text is long
                        chunks = self.create_chunks(cleaned_text)
                        source_stats['chunks_created'] += len(chunks)
                        processed_texts.extend(chunks)
            
            all_texts.extend(processed_texts)
            processing_stats['sources'][source] = source_stats
            
            logger.info(f"✅ {source}: {len(processed_texts)} final texts from {len(texts)} input texts")
        
        # Update global stats
        processing_stats['after_cleaning'] = sum(s['cleaned_count'] for s in processing_stats['sources'].values())
        processing_stats['after_quality_filter'] = sum(s['quality_passed'] for s in processing_stats['sources'].values())
        processing_stats['after_chunking'] = sum(s['chunks_created'] for s in processing_stats['sources'].values())
        
        # Remove duplicates
        logger.info("🔄 Removing duplicates...")
        unique_texts = self.remove_duplicates(all_texts)
        processing_stats['final_count'] = len(unique_texts)
        
        duplicate_count = len(all_texts) - len(unique_texts)
        if duplicate_count > 0:
            logger.info(f"🗑️ Removed {duplicate_count} duplicate texts")
        
        # Create dataset splits
        logger.info("📊 Creating train/validation splits...")
        dataset = self.create_dataset_splits(unique_texts)
        
        # Save processing statistics
        self.save_processing_stats(processing_stats)
        
        # Final validation
        self.validate_dataset(dataset)
        
        logger.info(f"✅ Preprocessing complete! Final dataset: {len(dataset['train'])} train, {len(dataset['test'])} validation")
        
        return dataset
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning."""
        
        if not text or not text.strip():
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove email addresses
        text = self.email_pattern.sub('', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub('', text)
        
        # Remove social media handles
        text = self.social_handle_pattern.sub('', text)
        
        # Fix repeated punctuation
        text = self.repeat_punct_pattern.sub(r'\1', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        text = ' '.join(text.split())  # Remove extra spaces
        
        # Fix common OCR/encoding issues
        text = self.fix_encoding_issues(text)
        
        # Normalize Nepali text
        text = self.normalize_nepali_text(text)
        
        return text.strip()
    
    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding and OCR issues in Nepali text."""
        
        # Common replacements for corrupted Nepali text
        replacements = {
            # Fix common OCR errors
            'अा': 'आ',
            'इी': 'ई', 
            'उू': 'ऊ',
            'एे': 'ए',
            'ओो': 'ओ',
            
            # Fix spacing around Devanagari
            ' ् ': '्',
            ' ं ': 'ं',
            ' ः ': 'ः',
            
            # Normalize quotation marks
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            
            # Fix common punctuation
            '|': '।',  # Pipe to Devanagari danda
            '||': '॥', # Double pipe to double danda
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def normalize_nepali_text(self, text: str) -> str:
        """Normalize Nepali text for consistency."""
        
        # Normalize Unicode (NFC form)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Ensure proper sentence ending
        if text and not any(text.endswith(p) for p in self.nepali_punctuation):
            text += '।'
        
        return text
    
    def passes_quality_check(self, text: str) -> bool:
        """Comprehensive quality check for Nepali text."""
        
        # Length check
        if len(text) < self.min_length or len(text) > self.max_length * 2:
            return False
        
        # Nepali script ratio check
        if not self.has_sufficient_nepali(text):
            return False
        
        # Language detection check
        if not self.is_nepali_language(text):
            return False
        
        # Repetition check
        if self.has_excessive_repetition(text):
            return False
        
        # Content quality check
        if not self.has_meaningful_content(text):
            return False
        
        # Sentence structure check
        if not self.has_proper_sentences(text):
            return False
        
        return True
    
    def has_sufficient_nepali(self, text: str) -> bool:
        """Check if text has sufficient Nepali (Devanagari) characters."""
        
        total_alpha = sum(1 for char in text if char.isalpha())
        if total_alpha == 0:
            return False
        
        nepali_chars = sum(1 for char in text if self.devanagari_range[0] <= char <= self.devanagari_range[1])
        nepali_ratio = nepali_chars / total_alpha
        
        return nepali_ratio >= self.min_nepali_ratio
    
    def is_nepali_language(self, text: str) -> bool:
        """Check if text is detected as Nepali language."""
        
        try:
            detected_lang = detect(text)
            return detected_lang == 'ne'
        except LangDetectException:
            # If detection fails, fall back to script analysis
            return self.has_sufficient_nepali(text)
    
    def has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        
        if len(text) == 0:
            return True
        
        # Count character frequencies
        char_counts = Counter(text)
        
        # Check for any character appearing too frequently
        for char, count in char_counts.items():
            if char.isalpha() and count / len(text) > self.max_repeat_ratio:
                return True
        
        # Check for repeated substrings
        words = text.split()
        if len(words) > 5:
            word_counts = Counter(words)
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            if repeated_words / len(words) > 0.3:
                return True
        
        return False
    
    def has_meaningful_content(self, text: str) -> bool:
        """Check if text contains meaningful content."""
        
        # Check for common Nepali words
        common_words = {
            'र', 'को', 'छ', 'हो', 'भएको', 'गर्ने', 'भन्ने', 'हुन्छ', 'नेपाल',
            'काठमाडौं', 'सरकार', 'जनता', 'देश', 'समाज', 'मान्छे', 'गए', 'भयो'
        }
        
        words = set(text.split())
        common_found = len(words & common_words)
        
        if common_found < 2:  # Need at least 2 common Nepali words
            return False
        
        # Check for reasonable word diversity
        unique_words = len(set(text.split()))
        total_words = len(text.split())
        
        if total_words > 0 and unique_words / total_words < 0.3:
            return False  # Too repetitive
        
        return True
    
    def has_proper_sentences(self, text: str) -> bool:
        """Check for proper sentence structure."""
        
        # Split by Nepali sentence delimiters
        sentences = [s.strip() for s in re.split(r'[।॥]', text) if s.strip()]
        
        if len(sentences) < self.min_sentence_count:
            return False
        
        # Check average sentence length
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:  # Very short sentences might be navigation/menu
                return False
        
        return True
    
    def create_chunks(self, text: str) -> List[str]:
        """Split long text into appropriate chunks."""
        
        if len(text) <= self.max_length:
            return [text]
        
        chunks = []
        
        # Split by paragraphs first (double danda or double newline)
        paragraphs = re.split(r'॥|\n\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph alone is too long, split by sentences
            if len(paragraph) > self.max_length:
                sentences = re.split(r'।', paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence += '।'  # Add back delimiter
                    
                    if len(current_chunk + sentence) <= self.max_length:
                        current_chunk += sentence
                    else:
                        if current_chunk and len(current_chunk) >= self.min_length:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            else:
                # Regular paragraph processing
                if len(current_chunk + paragraph + '॥') <= self.max_length:
                    current_chunk += paragraph + '॥'
                else:
                    if current_chunk and len(current_chunk) >= self.min_length:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '॥'
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_length:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts efficiently."""
        
        # Use set for exact duplicates
        seen = set()
        unique_texts = []
        
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        # For near-duplicates, use simple heuristic
        # (More sophisticated deduplication could use similarity metrics)
        final_texts = []
        
        for i, text in enumerate(unique_texts):
            is_duplicate = False
            
            # Check against recent texts (sliding window for efficiency)
            start_idx = max(0, i - 1000)  # Check last 1000 texts
            
            for j in range(start_idx, i):
                if self.are_near_duplicates(text, unique_texts[j]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_texts.append(text)
        
        return final_texts
    
    def are_near_duplicates(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """Check if two texts are near duplicates using Jaccard similarity."""
        
        # Quick length check
        if abs(len(text1) - len(text2)) / max(len(text1), len(text2)) > 0.3:
            return False
        
        # Jaccard similarity on word level
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return True
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return False
        
        jaccard_sim = intersection / union
        return jaccard_sim >= threshold
    
    def create_dataset_splits(self, texts: List[str]) -> DatasetDict:
        """Create train/validation splits."""
        
        # Shuffle texts
        import random
        random.shuffle(texts)
        
        # Create splits
        split_ratio = self.config.get('validation_split', 0.1)
        split_idx = int(len(texts) * (1 - split_ratio))
        
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Create datasets
        train_dataset = Dataset.from_dict({'text': train_texts})
        val_dataset = Dataset.from_dict({'text': val_texts})
        
        return DatasetDict({
            'train': train_dataset,
            'test': val_dataset  # Using 'test' as validation split
        })
    
    def save_processing_stats(self, stats: Dict[str, Any]):
        """Save detailed processing statistics."""
        
        stats_file = self.output_dir / "processing_stats.json"
        
        # Add timestamp and configuration
        stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        stats['config'] = self.config
        
        # Calculate processing efficiency
        if stats['total_input'] > 0:
            stats['efficiency'] = {
                'cleaning_rate': stats['after_cleaning'] / stats['total_input'],
                'quality_rate': stats['after_quality_filter'] / stats['total_input'],
                'final_rate': stats['final_count'] / stats['total_input']
            }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Processing statistics saved to {stats_file}")
    
    def validate_dataset(self, dataset: DatasetDict):
        """Final validation of processed dataset."""
        
        logger.info("🔍 Validating processed dataset...")
        
        for split_name, split_dataset in dataset.items():
            texts = split_dataset['text']
            
            # Basic statistics
            total_texts = len(texts)
            total_chars = sum(len(text) for text in texts)
            avg_length = total_chars / total_texts if total_texts > 0 else 0
            
            # Quality checks
            empty_texts = sum(1 for text in texts if not text.strip())
            short_texts = sum(1 for text in texts if len(text) < self.min_length)
            long_texts = sum(1 for text in texts if len(text) > self.max_length * 1.5)
            
            logger.info(f"📊 {split_name.upper()} split validation:")
            logger.info(f"  Total texts: {total_texts:,}")
            logger.info(f"  Average length: {avg_length:.1f} characters")
            logger.info(f"  Empty texts: {empty_texts}")
            logger.info(f"  Short texts (<{self.min_length}): {short_texts}")
            logger.info(f"  Long texts (>{self.max_length * 1.5:.0f}): {long_texts}")
            
            # Sample quality check
            if total_texts > 0:
                sample_size = min(100, total_texts)
                sample_texts = texts[:sample_size]
                
                nepali_count = sum(1 for text in sample_texts if self.has_sufficient_nepali(text))
                logger.info(f"  Nepali script ratio in sample: {nepali_count}/{sample_size} ({nepali_count/sample_size*100:.1f}%)")
        
        logger.info("✅ Dataset validation complete")
