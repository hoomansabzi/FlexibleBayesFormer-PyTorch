"""BayesianWindTunnels: task generators with exact Bayesian posteriors."""

from .AssociativeRecall import MultiHopRecallWindTunnel, OverwriteRecallWindTunnel, RecallWindTunnel
from .BayesianRegression import RegressionWindTunnel
from .Bijection import BijectionWindTunnel
from .HiddenMarkovModel import HMMWindTunnel

__all__ = ["BijectionWindTunnel", "RecallWindTunnel", "OverwriteRecallWindTunnel",
           "MultiHopRecallWindTunnel", "HMMWindTunnel", "RegressionWindTunnel"]
