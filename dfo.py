# Python
import numpy as np
import sys
import pickle
from pathlib import Path
import concurrent.futures
import sacrebleu

# Pytorch
import torch
from torchtext import data                                                     

# JoeyNMT
from joeynmt.helpers import load_config, set_seed, load_checkpoint, make_logger, log_cfg
from joeynmt.data import load_data, make_data_iter
from joeynmt.model import build_model

# In this package
from scripts.grad_estimators import GradEstimator
from scripts.reward_function import my_reward
from scripts.helpers import STEPSIZE_HELPER, serialize, subarray_generator



class DFTrainManager:
    """ Manages configuration, training and validation."""

    def __init__(self, config):
        
        # configurations
        self.train_cfg = config['training']
        self.model_cfg = config['model']
        self.data_cfg = config['data']

        # output
        self.output_dir = Path(self.train_cfg['model_dir'])
        self.output_dir.mkdir(parents=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints/'
        self.checkpoint_dir.mkdir()

        # logging, saves all console output to a file
        self.logger = make_logger(str(self.output_dir) + '/logging.txt')
        log_cfg(config, self.logger)

        # hyperparameters    
        self.iterations = self.train_cfg['iterations']
        self.batch_size = self.train_cfg['batch_size']
        self.use_cuda = self.train_cfg.get('use_cuda', False)
        self.train_embeddings = True   # might be later set to False by prep_model()
        self.num_workers = self.train_cfg.get('num_workers', 1)

        # general
        set_seed(seed=self.train_cfg.get("random_seed", 42))


    def _prep_model(self):
        """
        Builds and initialized the model as described in the configuration.
        The model can be initialized randomly or from a checkpoint. In addition, pre-trained embeddings can
        loaded from a different checkpoint. See configs/example.yaml for more info.
        """

        # initialize model weights according to model_cfg
        self.model = build_model(self.model_cfg, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)
        self.logger.info('Model loaded.')

        if self.train_cfg.get('start', False):

            # load all model weights from a given checkpoint                                                  
            self.model_start = load_checkpoint(path=self.train_cfg['start'], use_cuda=self.use_cuda) 
            self.model.load_state_dict(self.model_start['model_state'])

            self.logger.info('Model checkpoint loaded from {}.'.format(self.train_cfg['start']))

        if self.train_cfg.get('emb_start', False):

            # load only the embeddings from a given checkpoint
            emb_start = load_checkpoint(path=self.train_cfg['emb_start'], use_cuda=self.use_cuda)                                          
            pretrained_embeddings = {name:param for name,param in emb_start['model_state'].items() if '_embed' in name}                                                 
            self.model.load_state_dict(pretrained_embeddings, strict=False)                     
            self.train_embeddings = False
            self.get_embedding_idxs()        # prepare to set all embedding perturbations to 0

            self.logger.info('Embeddings loaded from {}.'.format(self.train_cfg['emb_start']))

        for param in self.model.parameters():
            param.requires_grad = False           # disable gradients for faster computation and lower memory usage

        if self.use_cuda:
            self.model.cuda()       # move all the model parameters to CUDA; might be redundant

        self.model.eval()    # disable dropout



    def _prep_data(self):
        """
        Loads train/dev/test data and initializes the training data iterator.
        """
        self.train_data, self.dev_data, self.test_data, self.src_vocab, self.trg_vocab = load_data(data_cfg=self.data_cfg)

        self.train_iter = data.BucketIterator(repeat=True, sort=False, dataset=self.train_data, batch_size=self.train_cfg['batch_size'],
                                              batch_size_fn=None, train=False, shuffle=True)

        self.logger.info('Data loaded.')


    def _prep_dfo(self):
        """
        Configures the DFO training process. 
        """

        self.theta = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.dim = len(self.theta)
        self.num_expl_directions = self.train_cfg['num_expl_directions']
        self.structured = self.train_cfg.get('structured', False)
        self.step_opt_type = self.train_cfg.get('step_opt_type', 'sgd')
        self.step_opt_params = self.train_cfg.get('step_opt_params', {'start': 0.001})

        if 'sigma' in self.model_start:            # if we're retraining or continuing from a DFO checkpoint, overwrite config                                       
            self.sigma = self.model_start['sigma']                                        
            self.step_opt = STEPSIZE_HELPER[self.model_start['step_opt_type']].load_state(self.model_start['step_opt_params'])
        else:
            self.sigma = self.train_cfg['sigma']['start']
            self.step_opt = STEPSIZE_HELPER[self.step_opt_type](self.dim, self.step_opt_params)

        if self.use_cuda:
            self.step_opt.cudafy()

        self.grad_estimator = GradEstimator(est_type=self.train_cfg.get('grad_estimator', 'antithetic'), 
                                            sigma=self.sigma, use_only_best=self.train_cfg.get('use_only_best', False),
                                            normalizer=self.train_cfg.get('normalizer', 'sigma'))

        self.logger.info('DFO configured.')
        
        # inform user about parallelization settings
        self.parallel_func_evals = self.train_cfg.get('parallel_func_evals', False)
        if self.num_workers > 1:
            self.logger.info("Using {} workers.".format(self.num_workers))
            if self.parallel_func_evals:
                self.logger.info("Parallelization type: parallel function evaluations")
            else: 
                self.logger.info("Parallelization type: parallel iterations")
        else:
            self.logger.info("Using 1 worker.")
       
        

    def train_and_validate(self):
        """
        Governs the actual training.

        :return list valid_rewards: The obtained validation rewards.
        :return list params_over_time: Contains every iteration's theta.
        :return list gradients: Contains the gradients computed in every iteration.
        :return list update_steps: Contains the update steps taken every iteration. 
        """

        # first, configure everything (order of calls is fixed!)
        self._prep_data()
        self._prep_model()
        self._prep_dfo()
        
        # set the validation frequency
        self.valid_freq = self.train_cfg['validation_freq']

        # keeps the highest validation score; used for saving checkpoints
        best_valid_score = 0.0

        # record some stuff
        params_over_time = [self.theta.clone()]
        valid_rewards = []
        gradients = []
        update_steps = []

        self.logger.info('Starting training...')
    
        for iteration in range(self.iterations + 1):

            if iteration % self.valid_freq == 0:
                # evaluate the validation set with the current model weights
                self.current_data = self.dev_data
                valid_reward = self.get_reward(self.theta)

                self.logger.info('Iteration: {}\t Reward valid. set: {}'.format(iteration, valid_reward))
                valid_rewards.append(valid_reward)

                # if a better reward was achieved, save a checkpoint
                if valid_reward > best_valid_score:
                    self.save_checkpoint(iteration)
                    best_valid_score = valid_reward

            for batch in iter(self.train_iter):
                
                # set to a class attribute so the reward function has direct access
                self.current_data = batch
                
                # generate the exploration directions
                expl_directions = None
                
                # if we're using multiple workers with each evaluating N expl. directions
                if self.num_workers > 1 and not self.parallel_func_evals:
                    
                    # generate enough expl. directions
                    expl_directions = self.get_expl_directions(self.num_expl_directions * self.num_workers)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor: 
                        
                        # give every worker N exploration directions to evaluate                               
                        futures = {executor.submit(self.collect_rewards, subset): i // self.num_expl_directions 
                                                   for subset, i in subarray_generator(expl_directions, self.num_expl_directions)}                                          
                        rewards = [0.0] * self.num_workers              
                        for f in concurrent.futures.as_completed(futures):    
                            rewards[futures[f]] = f.result() 
                    
                    # stack the rewards into one tensor  
                    rewards = torch.stack(rewards)

                    # prepare for the gradient estimation (needs to be a 1D array)
                    rewards = torch.cat((rewards[:, :self.num_expl_directions].flatten(),
                                         rewards[:, self.num_expl_directions:].flatten()))

                else:   # if we're doing parallel function evaluations or no parallelization at all

                    expl_directions = self.get_expl_directions(self.num_expl_directions)
                    rewards = self.collect_rewards(expl_directions)

                # estimate the gradient
                grad = self.grad_estimator.estimate(expl_directions, rewards)

                # update theta
                update_step = self.step_opt.update(self.theta, -grad)
                
                # keep track
                gradients.append(grad)
                update_steps.append(update_step)
                params_over_time.append(self.theta.clone())

                break    # we only 'see' one batch per iteration

        return valid_rewards, params_over_time, gradients, update_steps



    def collect_rewards(self, expl_directions):
        """
        Governs the reward function evaluations. 

        :param expl_directions: The noise vector added to theta to create candidates theta +- sigma*expl_direction
        :return torch.DoubleTensor: the collected rewards
        """
        
        # generates the candidates; this is specific to the gradient estimator used
        candidates = self.grad_estimator.get_theta_candidates(self.theta, self.sigma, expl_directions)
        
        # parallel execution of the function evaluations
        if self.parallel_func_evals:                                        
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:                                                            
                futures = {executor.submit(self.get_reward, candidate): i for i, candidate in enumerate(candidates)}                                          
                rewards = [0.0] * len(candidates)              
                for f in concurrent.futures.as_completed(futures):    
                    rewards[futures[f]] = f.result()                                                 
        else:                                                                  
            rewards = [self.get_reward(candidate) for candidate in candidates]                                                                                
        
        return torch.DoubleTensor(rewards).cuda() if self.use_cuda else torch.DoubleTensor(rewards)


    def get_expl_directions(self, size):
        """
        Generates noise/exploration directions, optionally orthogonal. If we're using pretrained embeddings, 
        all the embedding noise is set to 0.  

        :param size: how many exploration directions to generate
        :return torch.DoubleTensor: the exploration directions, a tensor size x self.dim
        """
        expl_directions = np.random.randn(size, self.dim)
            
        if self.structured:
            u, s, vh = np.linalg.svd(expl_directions, full_matrices=False)
            expl_directions = u @ vh
            
        if not self.train_embeddings:
            for idx_range in self.embedding_idxs:
                expl_directions[:, idx_range] = 0.0

        return torch.from_numpy(expl_directions).cuda() if self.use_cuda else torch.from_numpy(expl_directions)

    
    def get_embedding_idxs(self):
        """
        Goes over the model parameters and notes the ranges of all embeddings. This is necessary when 
        we're using pretrained embeddings, since we don't want to change them during training.
        """
        self.embedding_idxs = []
        running_idx = 0

        for name, param in self.model.named_parameters():
            size = param.numel()
            if '_embed' in name:
                self.embedding_idxs.append(range(running_idx, running_idx + size))
            running_idx += size


    def get_reward(self, **kwargs):
        """
        A placeholder for the reward function.
        """
        raise NotImplementedError


    def save_checkpoint(self, iteration):
        """
        Saves a DFO checkpoint. This includes the weights, sigma, and the optimizer.
        """

        # update the model to match theta
        torch.nn.utils.vector_to_parameters(self.theta, self.model.parameters())

        state = {
            "model_state": self.model.state_dict(),
            "sigma": self.sigma,
            "step_opt_type": self.step_opt_type,
            "step_opt_params": self.step_opt.get_state()
        }

        torch.save(state, str(self.checkpoint_dir) + '/{}.pt'.format(iteration))
        self.logger.info('Saved checkpoint for iteration {}.'.format(iteration))



if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='The YAML configuration file.')
    parser.add_argument('--debug', action='store_true', help='If set, \
                         parameters, gradients and update steps are also saved.')
    args = parser.parse_args()

    DFTrainManager.get_reward = my_reward
    config = load_config(args.config_file)

    tm = DFTrainManager(config)

    valid_rewards, params_over_time, gradients, update_steps = tm.train_and_validate()

    serialize(valid_rewards, str(tm.output_dir) + '/valid_rewards')

    if args.debug:
        serialize(params_over_time, str(tm.output_dir) + '/params_over_time')
        serialize(gradients, str(tm.output_dir) + '/gradients')
        serialize(update_steps, str(tm.output_dir) + '/update_steps')
    
