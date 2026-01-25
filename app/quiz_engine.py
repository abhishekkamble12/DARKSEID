"""
Hugging Face Quiz Generator Engine
===================================
Uses local/Colab Hugging Face models to generate quizzes from text.
Supports 4-bit quantization for memory efficiency.

Usage:
    from quiz_engine import QuizGenerator
    
    generator = QuizGenerator()  # Loads model once
    quiz = generator.generate(text="Your content here", num_questions=5)
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if HuggingFace libraries are available
HF_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HF_AVAILABLE = True
except ImportError:
    logger.warning(
        "HuggingFace libraries not installed. "
        "Install with: pip install -r requirements-hf.txt"
    )


# System prompt tuned for Gemma 2's instruction format
HF_SYS_PROMPT = """<start_of_turn>user
You are a strict JSON Output Generator.
Task: Convert the provided text into a Multiple Choice Quiz.

Rules:
1. Output MUST be raw JSON. No markdown (```json), no conversational filler.
2. Create {num_questions} questions.
3. 'correct_answer' must be the exact string text of the correct option.
4. Focus on conceptual "Why" and "How" questions, not just "What" facts.

JSON Schema:
{{
  "quiz_title": "string",
  "questions": [
    {{
      "id": 1,
      "question": "string",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct_answer": "string (exact text of correct option)",
      "explanation": "string"
    }}
  ]
}}

Text to process:
{context_text}
<end_of_turn>
<start_of_turn>model
{{"""


class QuizGenerator:
    """
    Generates quizzes using a local Hugging Face model.
    
    The model is loaded once during initialization and reused for all requests.
    Uses 4-bit quantization by default to reduce memory usage.
    
    Attributes:
        model_name: The HuggingFace model ID to use
        model: The loaded model instance
        tokenizer: The loaded tokenizer instance
        device: The device the model is loaded on
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        use_4bit: bool = True,
        device_map: str = "auto",
        max_memory: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the Quiz Generator with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model ID. Default is gemma-2-2b-it (smaller).
                       Use 'google/gemma-2-9b-it' for better quality if you have VRAM.
            use_4bit: Whether to use 4-bit quantization (saves ~75% memory)
            device_map: Device placement strategy ('auto', 'cuda', 'cpu')
            max_memory: Optional memory limits per device, e.g. {0: "6GB"}
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace libraries not available. "
                "Install with: pip install transformers torch bitsandbytes accelerate"
            )
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        
        logger.info(f"Loading model: {model_name}")
        
        # Configure quantization
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info("Using 4-bit quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
    
    def _clean_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from model output.
        
        Handles cases where the model includes extra text or markdown formatting.
        
        Args:
            raw_text: Raw model output that may contain JSON
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        if not raw_text:
            return None
        
        # Try to find JSON object between { and }
        # Use regex to find the outermost JSON object
        json_patterns = [
            r'\{[\s\S]*\}',  # Match outermost braces
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',  # Generic code block
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, raw_text)
            if match:
                json_str = match.group(1) if match.lastindex else match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Last resort: try parsing the whole thing
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from: {raw_text[:200]}...")
            return None
    
    def _format_quiz_for_frontend(self, quiz_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format quiz data to be compatible with the frontend Quiz Card component.
        
        Transforms the quiz to match the expected format:
        {
            "cards": [
                {
                    "question": "...",
                    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                    "answer": "...",
                    "explanation": "...",
                    "tts_text": "Question: [question text]. Think carefully."
                }
            ]
        }
        
        Args:
            quiz_data: Raw quiz data from the model
            
        Returns:
            Formatted quiz data for frontend
        """
        if not quiz_data:
            return {"cards": []}
        
        cards = []
        questions = quiz_data.get("questions", [])
        
        for q in questions:
            card = {
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "answer": q.get("correct_answer", ""),
                "explanation": q.get("explanation", ""),
                "tts_text": f"Question: {q.get('question', '')}. Think carefully about the options."
            }
            cards.append(card)
        
        return {
            "quiz_title": quiz_data.get("quiz_title", "Generated Quiz"),
            "cards": cards
        }
    
    def generate(
        self,
        text: str,
        num_questions: int = 5,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        format_for_frontend: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a quiz from the provided text.
        
        Args:
            text: The source text to generate questions from
            num_questions: Number of questions to generate (default: 5)
            max_new_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (default: 0.7)
            format_for_frontend: Whether to format output for frontend (default: True)
            
        Returns:
            Dictionary containing the quiz data, or None if generation fails
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded")
            return None
        
        # Truncate text if too long (roughly 3000 chars for safety)
        if len(text) > 3000:
            text = text[:3000] + "..."
            logger.warning("Input text truncated to 3000 characters")
        
        # Build the prompt
        prompt = HF_SYS_PROMPT.format(
            num_questions=num_questions,
            context_text=text
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        logger.info(f"Generating {num_questions} questions...")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # The prompt ends with "{" so we need to prepend it
        generated_text = "{" + generated_text
        
        # Parse and clean the JSON
        quiz_data = self._clean_json(generated_text)
        
        if quiz_data and format_for_frontend:
            return self._format_quiz_for_frontend(quiz_data)
        
        return quiz_data
    
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None


# Global instance placeholder (initialized lazily)
_quiz_generator: Optional[QuizGenerator] = None


def get_quiz_generator(
    model_name: str = "google/gemma-2-2b-it",
    **kwargs
) -> Optional[QuizGenerator]:
    """
    Get or create the global QuizGenerator instance.
    
    This ensures the model is only loaded once and reused across requests.
    
    Args:
        model_name: HuggingFace model ID
        **kwargs: Additional arguments passed to QuizGenerator
        
    Returns:
        QuizGenerator instance or None if HF libraries aren't available
    """
    global _quiz_generator
    
    if not HF_AVAILABLE:
        logger.warning("HuggingFace libraries not available")
        return None
    
    if _quiz_generator is None:
        try:
            _quiz_generator = QuizGenerator(model_name=model_name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize QuizGenerator: {e}")
            return None
    
    return _quiz_generator


def generate_quiz_from_text(
    text: str,
    num_questions: int = 5,
    use_local_model: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to generate a quiz from text.
    
    Args:
        text: Source text to generate questions from
        num_questions: Number of questions to generate
        use_local_model: Whether to use the local HF model (if available)
        
    Returns:
        Quiz dictionary or None if generation fails
    """
    if use_local_model and HF_AVAILABLE:
        generator = get_quiz_generator()
        if generator:
            return generator.generate(text=text, num_questions=num_questions)
    
    return None


# For testing
if __name__ == "__main__":
    # Test the generator
    test_text = """
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience without being explicitly programmed. 
    It focuses on developing algorithms that can access data and use it to learn 
    for themselves. The process begins with observations or data, such as examples, 
    direct experience, or instruction, to look for patterns in data and make better 
    decisions in the future. The primary aim is to allow computers to learn 
    automatically without human intervention and adjust actions accordingly.
    
    There are three main types of machine learning:
    1. Supervised Learning - The algorithm learns from labeled training data
    2. Unsupervised Learning - The algorithm finds patterns in unlabeled data  
    3. Reinforcement Learning - The algorithm learns through trial and error
    """
    
    print("Testing QuizGenerator...")
    generator = get_quiz_generator()
    
    if generator:
        quiz = generator.generate(test_text, num_questions=3)
        if quiz:
            print(json.dumps(quiz, indent=2))
        else:
            print("Failed to generate quiz")
    else:
        print("QuizGenerator not available (check HF dependencies)")
