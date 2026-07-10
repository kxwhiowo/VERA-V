from .base_judge import BaseJudge
import torch
import torch.nn.functional as F


class JudgeWhiteBox(BaseJudge):
    def __init__(self, *args, **kwargs):
        pass

    def batch_eval(self, questions, responses, target_lm, target_tokenizer, *args, **kwargs):
        # Concatenate questions and responses to form full Q&A pairs
        q_a_pairs = [q + " " + r for q, r in zip(questions, responses)]
        
        # Tokenize the full Q&A pairs
        tokenized_q_a_pairs = target_tokenizer(
            q_a_pairs, return_tensors="pt", padding=True, truncation=False
        ).to(target_lm.device)
        
        # Tokenize only the responses separately to identify where they start
        tokenized_responses = target_tokenizer(
            responses, return_tensors="pt", padding=True, truncation=False
        ).to(target_lm.device)

        # Compute model logits (logits.shape: [batch_size, seq_length, vocab_size])
        with torch.no_grad():
            outputs = target_lm(**tokenized_q_a_pairs)
        logits = outputs.logits  # [B, seq_len, vocab_size]

        # Compute log probabilities over the vocabulary
        log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

        # Identify response token positions in the full Q&A tokenization
        response_token_ids = tokenized_responses["input_ids"]  # [B, resp_len]
        response_start_positions = (tokenized_q_a_pairs["input_ids"] == response_token_ids[:, 0].unsqueeze(-1)).int().argmax(dim=1)

        # Extract log probabilities of actual response tokens
        batch_size, resp_len = response_token_ids.shape
        log_likelihoods = torch.zeros(batch_size, device=log_probs.device)

        for i in range(batch_size):
            start_idx = response_start_positions[i]
            end_idx = start_idx + resp_len  # Ensure it aligns with the response length

            # Get log probabilities corresponding to response tokens
            target_token_probs = log_probs[i, start_idx:end_idx, :]  # [resp_len, vocab_size]
            actual_token_log_probs = target_token_probs.gather(1, response_token_ids[i].unsqueeze(-1)).squeeze(-1)  # [resp_len]

            # Sum log probabilities to get total log likelihood
            log_likelihoods[i] = actual_token_log_probs.sum()

        return log_likelihoods  # Tensor of shape [B]


