from langchain_core.tools import tool
import re
from typing import Dict, List, Any, Optional
from collections import Counter
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 # Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
class DocumentAnalyzerTool:
    """Advanced tool for analyzing document content."""
    
    def __init__(self):
        """Initialize the document analyzer tool."""
        self.stop_words = set(stopwords.words('english'))
    
    @tool("analyze_document")
    def analyze(self, text: str) -> str:
        """
        Comprehensively analyze document content to extract key information.
        
        Args:
            text: The document text to analyze
            
        Returns:
            A string containing detailed analysis results
        """
        try:
            if not text or len(text.strip()) == 0:
                return "Error: Empty document provided."
            
            # Basic statistics
            word_count = len(re.findall(r'\w+', text))
            sentence_count = len(sent_tokenize(text))
            paragraph_count = len(text.split('\n\n'))
            avg_sentence_length = word_count / max(1, sentence_count)
            
            # Extract key phrases using simple TF-IDF approach
            key_phrases = self._extract_key_phrases(text)
            
            # Simple sentiment analysis
            sentiment_score, sentiment = self._analyze_sentiment(text)
            
            # Extract potential entities (names, organizations, etc.)
            entities = self._extract_entities(text)
            
            # Format the results
            analysis = f"""
 Document Analysis Report
 ========================
 Basic Statistics:----------------
 Word count: {word_count}
 Sentence count: {sentence_count}
 Paragraph count: {paragraph_count}
 Average sentence length: {avg_sentence_length:.1f} words
 Content Summary:--------------
Key phrases:
 """
            
            # Add key phrases
            for i, (phrase, score) in enumerate(key_phrases[:5], 1):
                analysis += f"{i}. {phrase} (relevance: {score:.2f})\n"
            
            # Add sentiment analysis
            analysis += f"\nSentiment: {sentiment} (score: {sentiment_score:.2f})\n"
            
            # Add entities if found
            if entities:
                analysis += "\nPotential entities:\n"
                for i, entity in enumerate(entities[:10], 1):
                    analysis += f"{i}. {entity}\n"
            
            # Add key sentences (simple extractive summarization)
            key_sentences = self._extract_key_sentences(text, key_phrases)
            if key_sentences:
                analysis += "\nKey content:\n"
                for i, sentence in enumerate(key_sentences[:3], 1):
                    analysis += f"{i}. {sentence}\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing document: {str(e)}"
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[tuple]:
        """
        Extract key phrases from text using a simple TF-IDF like approach.
        
        Args:
            text: The document text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of (phrase, score) tuples
        """
        # Tokenize and clean words
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Calculate "importance" score (simplified TF-IDF)
        word_scores = {}
        for word, count in word_freq.items():
            # Higher score for less common words with multiple occurrences
            word_scores[word] = count * math.log(len(words) / (count + 1) + 1)
        
        # Extract phrases (1-3 words)
        sentences = sent_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            
            # Single words
            for i in range(len(sentence_words)):
                word = sentence_words[i]
                if word.isalnum() and word not in self.stop_words:
                    phrases.append((word, word_scores.get(word, 0)))
            
            # Bigrams (2-word phrases)
            for i in range(len(sentence_words) - 1):
                if (sentence_words[i].isalnum() and 
                    sentence_words[i+1].isalnum() and
                    sentence_words[i] not in self.stop_words):
                    bigram = f"{sentence_words[i]} {sentence_words[i+1]}"
                    # Score is average of individual word scores plus bonus for multi-word
                    score = (word_scores.get(sentence_words[i], 0) + 
                             word_scores.get(sentence_words[i+1], 0)) / 2 * 1.5
                    phrases.append((bigram, score))
        
        # Sort by score and return top phrases
        sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and subphrases
        unique_phrases = []
        for phrase, score in sorted_phrases:
            if len(unique_phrases) >= max_phrases:
                break
            
            # Check if this phrase is a subphrase of any existing phrases
            is_subphrase = any(phrase in existing for existing, _ in unique_phrases)
            
            if not is_subphrase:
                unique_phrases.append((phrase, score))
        
        return unique_phrases
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """
        Perform basic sentiment analysis.
        
        Args:
            text: The document text
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        # Simple lexicon-based approach
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'best', 'innovative',
            'impressive', 'helpful', 'beneficial', 'advantage', 'success',
            'happy', 'pleased', 'effective', 'useful', 'better', 'remarkable'
        ]
        
        negative_words = [
            'bad', 'poor', 'negative', 'worst', 'problem', 'issue',
            'disappointing', 'difficult', 'failure', 'concern', 'weakness',
            'disadvantage', 'trouble', 'ineffective', 'useless', 'worse'
        ]
        
        # Count positive and negative words
        words = word_tokenize(text.lower())
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total = pos_count + neg_count
        if total == 0:
            score = 0
        else:
            score = (pos_count - neg_count) / total
        
        # Map score to sentiment label
        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return score, sentiment
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential named entities using simple heuristics.
        
        Args:
            text: The document text
            
        Returns:
            List of potential entity strings
        """
        # This is a simplified approach - in production, use a proper NER model
        
        # Find capitalized words that aren't at the start of sentences
        sentences = sent_tokenize(text)
        entities = set()
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            # Skip the first word of the sentence
            for i in range(1, len(words)):
                word = words[i]
                
                # Check if word is capitalized and alphabetic
                if (word[0].isupper() and word.isalpha() and 
                    word.lower() not in self.stop_words and len(word) > 1):
                    
                    # Check for multi-word entities
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        entities.add(f"{word} {words[i+1]}")
                    else:
                        entities.add(word)
        
        return list(entities)
    
    def _extract_key_sentences(self, text: str, key_phrases: List[tuple]) -> List[str]:
        """
        Extract the most important sentences based on key phrases.
        
        Args:
            text: The document text
            key_phrases: List of (phrase, score) tuples
            
        Returns:
            List of important sentences
        """
        sentences = sent_tokenize(text)
        sentence_scores = []
        
        phrase_dict = {phrase: score for phrase, score in key_phrases}
        
        for sentence in sentences:
            score = 0
            
            # Score based on presence of key phrases
            for phrase, phrase_score in key_phrases:
                if phrase.lower() in sentence.lower():
                    score += phrase_score
            
            # Adjust score based on sentence length (prefer medium-length sentences)
            words = len(word_tokenize(sentence))
            if words < 5:
                score *= 0.5  # Penalize very short sentences
            elif words > 25:
                score *= 0.8  # Slightly penalize very long sentences
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sorted_sentences[:5]]