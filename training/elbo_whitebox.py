from torch import nn
from .base_loss import BaseLoss


class ELBOWhiteBox(BaseLoss):
    def __init__(self, target_lm, judge, device, queries_to_use="original"):
        super().__init__(target_lm, judge, device)
        self.queries_to_use = queries_to_use


    def forward(self, adversarial_queries, original_queries, log_likelihoods):
        self.model = self.target_lm.model
        self.tokenizer = self.target_lm.tokenizer
        # calculate the weighting for eahc query
        responses = self.target_lm.batch_respond(adversarial_queries)
        queries = adversarial_queries
        # make sure these are on the same device
        judge_scores = self.judge.batch_eval(queries, 
                                             responses, 
                                             self.model, 
                                             self.tokenizer).to(self.attacker_device)

        weights = judge_scores - log_likelihoods
        loss = weights * log_likelihoods * (1 / len(queries))

        loss_dct = {
            "weights_mean": weights.mean().item(),
            "weights_std": weights.std().item(),
            "weights_max": weights.max().item(),
            "weights_min": weights.min().item(),
            "log_likelihood_mean": log_likelihoods.mean().item(),
            "log_likelihood_std": log_likelihoods.std().item(),
            "log_likelihood_max": log_likelihoods.max().item(),
            "log_likelihood_min": log_likelihoods.min().item(),
            "judge_score_mean": judge_scores.mean().item(),
            "judge_score_std": judge_scores.std().item(),
            "judge_score_max": judge_scores.max().item(),
            "judge_score_min": judge_scores.min().item(),
            "loss": -1 * loss.sum(),
            "responses": responses,
        }
        return loss_dct
