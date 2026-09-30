"""Pair grouping and fixed final-state linear-head checks."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from neurotopo.final_probe import final_hidden, score_final
from neurotopo.models import RecurrentModel
from neurotopo.task import INPUT_SIZE
from neurotopo.transfer_probe import fit_prior_head, split_batch, validate_splits


def test_final_state_matches_original_readout_and_pair_split():
    torch.manual_seed(15)
    model=RecurrentModel(INPUT_SIZE,64,4,groups=4,topology="block")
    batch=split_batch(8,4,1010000)
    hidden=final_hidden(model,batch)
    _,original=model(batch.x,batch.pre_index)
    assert hidden.shape==(16,64)
    assert torch.equal(model.readout(hidden),original)
    assert torch.equal(batch.x[::2,:batch.pre_index+1],batch.x[1::2,:batch.pre_index+1])
    assert not torch.equal(hidden[::2],hidden[1::2])
    validate_splits({"train_a":batch,"train_b":split_batch(8,4,1015000),
                     "validation":split_batch(8,4,1020000),"test":split_batch(8,4,1030000)})
    with pytest.raises(ValueError,match="duplicate prefix"):
        validate_splits({"train_a":batch,"test":batch})


def test_final_ridge_reads_both_counterfactual_siblings():
    rng=np.random.default_rng(3)
    targets=torch.from_numpy(rng.integers(4,size=256))
    state=torch.zeros((256,64))
    state[:,:4]=F.one_hot(targets,4).float()
    mean=state[:192].double().mean(0)
    scale=state[:192].double().std(0,unbiased=False).clamp_min(.001)
    weight=fit_prior_head(state[:192],targets[:192],mean,scale,1)
    result=score_final(state[192:],targets[192:],mean,scale,weight)
    assert weight.shape==(65,4)
    assert result["revise_accuracy"]==result["confirm_accuracy"]==1.0
    assert result["both_pair_accuracy"]==1.0
