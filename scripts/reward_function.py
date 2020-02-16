import copy
import torch
from torchtext.data import Dataset, Batch
from joeynmt.prediction import validate_on_data
from joeynmt.batch import Batch as JoeyBatch
from scripts.helpers import EVAL_HELPER


def my_reward(tm, candidate_theta):
    """
    Computes the reward of a single candidate-theta.

    :param candidate_theta: the candidate model parameters
    :return reward: a scalar representing the reward of candidate_theta on the current data
    """
    
    # set the evaluation/reward metric
    eval_metric = tm.train_cfg.get("eval_metric", "bleu")

    dummy_model = copy.deepcopy(tm.model) # make a copy of the model, mostly for parallelization scenarios

    # set candidate_theta to be the model parameters
    torch.nn.utils.vector_to_parameters(candidate_theta, dummy_model.parameters())  
    
    # tm.current_data is set either to a Dataset, or a Batch instance

    if isinstance(tm.current_data, Dataset):  # if we're evaluating an entire dataset, a function already exists in JoeyNMT
        return validate_on_data(dummy_model, tm.current_data, batch_size=tm.batch_size,
                                use_cuda=tm.use_cuda, max_output_length=tm.train_cfg.get("max_output_length", None),
                                level=tm.data_cfg["level"], eval_metric=eval_metric, logger=tm.logger)[0]
    
    elif isinstance(tm.current_data, Batch):   # if we're evaluating a single batch, like during training

        batch = JoeyBatch(tm.current_data, dummy_model.pad_index, use_cuda=tm.use_cuda)
        sort_reverse_index = batch.sort_by_src_lengths()     

        output, _ = dummy_model.run_batch(batch=batch, beam_size=0, beam_alpha=-1,
                                    max_output_length=tm.train_cfg.get("max_output_length", None))

        output = output[sort_reverse_index]
        target = batch.trg[sort_reverse_index]
        
        decoded_output = dummy_model.trg_vocab.arrays_to_sentences(arrays=output,
                                                                cut_at_eos=True)
        decoded_target = dummy_model.trg_vocab.arrays_to_sentences(arrays=target,
                                                                cut_at_eos=True)
        references = [" ".join(t) for t in decoded_target]
        hypothesis = [" ".join(t) for t in decoded_output]
        
        reward = EVAL_HELPER[eval_metric](hypothesis, references)

        return reward

    else:
        raise TypeError("Inputs should be either a Batch or a Dataset instance.")
