from judge.base_judge import BaseJudge
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class LlamaGuard2Judge(BaseJudge):
    """
    Llama Guard 2 judge implementation for safety evaluation.
    Uses the Meta-Llama-Guard-2-8B model to classify responses as safe or unsafe.
    """
    
    def __init__(self, device: str = "cuda", model_name: str = "meta-llama/Meta-Llama-Guard-2-8B"):
        """
        Initialize Llama Guard 2 judge.
        
        Args:
            device: Device to run the model on (e.g., "cuda", "cpu")
            model_name: HuggingFace model ID for Llama Guard 2
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        
        # Set x_model flag for base class compatibility
        self.x_model = True
    
    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Implementation of the abstract get_response method from BaseJudge.
        
        Args:
            system_prompt: System prompt (not used in Llama Guard 2)
            user_prompt: User prompt to evaluate
            
        Returns:
            str: Safety assessment result from Llama Guard 2
        """
        # For Llama Guard 2, we treat the user_prompt as the response to evaluate
        # and use a default prompt as the original query
        return self.judge_response("Evaluate this content for safety", user_prompt)
        
    def judge_response(self, original_prompt: str, response_text: str) -> str:
        """
        Judge a single response using Llama Guard 2.
        
        Args:
            original_prompt: The original user query
            response_text: The model's response to evaluate
            
        Returns:
            str: Safety assessment result from Llama Guard 2
        """
        chat = [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": response_text},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=100, 
                pad_token_id=self.tokenizer.pad_token_id or 0,
                do_sample=False
            )
        
        prompt_len = input_ids.shape[-1]
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return result.strip()
    
    def is_harmful(self, original_prompt: str, response_text: str) -> bool:
        """
        Check if a response is deemed harmful by Llama Guard 2.
        
        Args:
            original_prompt: The original user query
            response_text: The model's response to evaluate
            
        Returns:
            bool: True if response is harmful, False otherwise
        """
        result = self.judge_response(original_prompt, response_text)
        return "unsafe" in result.lower()
    
    def evaluate_batch(self, prompts: List[str], responses: List[str]) -> List[str]:
        """
        Batch evaluation that returns only safe/unsafe classifications.
        
        Args:
            prompts: List of original prompts/queries
            responses: List of responses to evaluate
            
        Returns:
            List[str]: List containing "safe" or "unsafe" for each response
        """
        results = []
        
        for prompt, response in zip(prompts, responses):
            # Get the raw assessment from Llama Guard 2
            assessment = self.judge_response(prompt, response)
            
            # Extract safety status
            if "unsafe" in assessment.lower():
                results.append("unsafe")
            else:
                results.append("safe")
        
        return results


def evaluate_safety_batch_llama(prompts: List[str], responses: List[str], device: str = "cuda") -> List[str]:
    """
    Standalone function for batch safety evaluation using Llama Guard 2.
    
    Args:
        prompts: List of original prompts/queries
        responses: List of responses to evaluate
        device: Device to run the model on
        
    Returns:
        List[str]: List containing "safe" or "unsafe" for each response
    """
    judge = LlamaGuard2Judge(device=device)
    return judge.evaluate_batch(prompts, responses)
            