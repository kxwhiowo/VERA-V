from torch import nn
import pdb
from .base_loss import BaseLoss
import torch

class ELBO(BaseLoss):
    """
        log_judge_scores = log P_LM(y*|x)
        log_likelihoods = log q_theta(x)
        log_prior = based model
    """
    def __init__(self, target_lm, judge, device,  entropy_coef=0.08,
                 kl_coef=0.05, baseline_momentum=0.95,queries_to_use="original"):
        super().__init__(target_lm, judge, device)
        self.queries_to_use = queries_to_use
        self.H = entropy_coef
        self.KL = kl_coef
        self.b = 0.0
        self.mom = baseline_momentum

    def forward(self, adversarial_queries, original_queries, log_likelihoods, entropies, log_prior):

        # calculate the weighting for eahc query
        responses = self.target_lm.batch_respond(adversarial_queries)
        # print("ELBO Responses:",responses)
        if self.queries_to_use == "original":
            queries = original_queries

        elif self.queries_to_use == "adversarial":
            queries = adversarial_queries
        # make sure these are on the same device
        
        with torch.no_grad():
            log_judge_scores = self.judge.batch_judge_train(queries, responses).to(
                self.attacker_device
            )
        # del responses
        actual_judge_scores = log_judge_scores.exp()
        print("**0.5 judge scores:",actual_judge_scores)
        # print("log_judge_scores",log_judge_scores)
        print("log likelihood:",log_likelihoods)
        # print("log prior:",log_prior)
        # print("Entropy:",entropies)
        # log_likelihoods = torch.clamp(log_likelihoods, min=-2, max=0)
        weights = (log_judge_scores-log_likelihoods)#.detach()
        # w = (weights - weights.mean()) / (weights.std() + 1e-8) 
        # weights=torch.relu(weights)
        # self.b = 0.95 * self.b + 0.05 * weights.mean()
        # -self.b).detach()
        pg = -1 * (weights.detach())* log_likelihoods 
        with torch.no_grad():
            pg_mag = (weights * (-log_likelihoods)).abs().mean()     # |PG|
            ent_mag = entropies.mean()                 # token entropy
        gamma = pg_mag / ent_mag                      # first‑batch match

        # ent_term = -gamma * entropies
        target_H = 1.2          # bits per generated token
        gamma += 0.02 * (entropies.mean().item() - target_H)
        gamma  = max(gamma, 0.0)
        ent= -gamma * self.H 
        # ent = -self.H * entropies
        self.H *= 1.005
        kl  =  self.KL * (log_likelihoods - log_prior)
        print("ENT:",ent)
        loss = (pg + ent + kl).mean()

        loss_dct = {
            "weights_mean": weights.mean().item(),
            "weights_std": weights.std().item(),
            "weights_max": weights.max().item(),
            "weights_min": weights.min().item(),
            "entropy": entropies.mean().item(),
            "kl": (log_likelihoods - log_prior).mean().item(),
            "log_likelihood_mean": log_likelihoods.mean().item(),
            "log_likelihood_std": log_likelihoods.std().item(),
            "log_likelihood_max": log_likelihoods.max().item(),
            "log_likelihood_min": log_likelihoods.min().item(),
            "judge_score_mean": actual_judge_scores.mean().item(),
            "judge_score_std": actual_judge_scores.std().item(),
            "judge_score_max": actual_judge_scores.max().item(),
            "judge_score_min": actual_judge_scores.min().item(),
            "loss": loss,
            "responses": responses,
            "judge_scores": actual_judge_scores.cpu().tolist(),
        }
        return loss_dct
