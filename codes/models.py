import os
import logging
import numpy as np
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...
      
    @abstractmethod
    def normalize_embedding(self):
        ...
        

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            
            if self.name in ['KG2E_KL', 'KG2E_EL']:
                head_v = torch.index_select(
                    self.entity_cov,
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)

                relation_v = torch.index_select(
                    self.relation_cov,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                tail_v = torch.index_select(
                    self.entity_cov,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
            if self.name in ['KG2E_KL', 'KG2E_EL']:
                head_v = torch.index_select(
                    self.entity_cov,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                relation_v = torch.index_select(
                    self.relation_cov,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                tail_v = torch.index_select(
                    self.entity_cov,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            if self.name in ['KG2E_KL', 'KG2E_EL']:
                head_v = torch.index_select(
                self.entity_cov,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation_v = torch.index_select(
                self.relation_cov,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail_v = torch.index_select(
                self.entity_cov,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        # return scores
        if self.name in ['KG2E_KL', 'KG2E_EL']:
            return self.func(head, relation, tail, batch_type, head_v, relation_v, tail_v)
        else:
            return self.func(head, relation, tail, batch_type)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
        
        if model.name in ['KG2E_KL', 'KG2E_EL']:
                        model.normalize_embedding()

        # negative scores
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    
                    if model.name in ['KG2E_KL', 'KG2E_EL']:
                        model.normalize_embedding()

                    score = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics


class ModE(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma):
        super(ModE, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def func(self, head, rel, tail, batch_type):
        score = self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)
        
        return score
    
    def normalize_embedding(self):
        ...


class HAKE(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        #nn.init.zeros_(
         #   tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        #)

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846


    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation = torch.chunk(rel, 2, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        #bias_relation = torch.clamp(bias_relation, max=1)
        #indicator = (bias_relation < -mod_relation)
        #bias_relation[indicator] = -mod_relation[indicator]

        #r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        r_score = mod_head * mod_relation - mod_tail

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        score = self.gamma.item() - (phase_score + r_score)
        print(score)
        
        return score
    
    def normalize_embedding(self):
        ...
    
class RotatE(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma):
        super(RotatE, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.pi = 3.14159262358979323846


    def func(self, head, rel, tail, batch_type):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = rel/(self.embedding_range.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if batch_type == BatchType.HEAD_BATCH:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        
        return score
    
    def normalize_embedding(self):
        ...
    
class TransE(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma):
        super(TransE, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )


    def func(self, head, rel, tail, batch_type):
        if batch_type == BatchType.HEAD_BATCH:
            score = head + (rel - tail)
        else:
            score = (head + rel) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        
        return score
    
    def normalize_embedding(self):
        ...
    
class UM(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma):
        super(UM, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim), requires_grad=False)


    def func(self, head, rel, tail, batch_type):
        score = head - tail
        
        score = torch.norm(score, p=2, dim=2)

        score = self.gamma.item() - score*score
        
        return score
    
    def normalize_embedding(self):
        ...
  

class KG2E_KL(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma, cmin, cmax):
        super(KG2E_KL, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.cmin = cmin
        self.cmax = cmax
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.entity_cov = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=self.cmin,
            b=self.cmax
        )

        self.relation_cov = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=self.cmin,
            b=self.cmax
        )
       
    
    def func(self, head, rel, tail, batch_type, head_v, rel_v, tail_v):
        """
        Calculate similarity based on KL divergence
        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
            = \frac{1}{2} \left(
                tr(\Sigma_r^{-1}\Sigma_e)
                + (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
                - \log \frac{det(\Sigma_e)}{det(\Sigma_r)} - k_e
            \right)
        """
        eps = 1.0e-10
        mu_e = head - tail
        sigma_e = head_v - tail_v
        mu_r = rel
        sigma_r = torch.clamp_min(rel_v, min=eps)
        
        #: a = tr(\Sigma_r^{-1}\Sigma_e)
        a = torch.sum(sigma_e / sigma_r, dim = 2)
        
        #print('a = ', a)
        
        #: b = (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
        """
        if batch_type == BatchType.HEAD_BATCH:
            b = torch.sum((mu_r - mu_e) ** 2 / sigma_r, dim = 2)
        else:
            b = torch.sum((mu_e - mu_r) ** 2 / sigma_r, dim = 2)
        """
        b = torch.sum((mu_r - mu_e) ** 2 / sigma_r, dim = 2)
        #print('b = ', b)
        
        #: c = \log \frac{det(\Sigma_e)}{det(\Sigma_r)}
        # = sum log (sigma_e)_i - sum log (sigma_r)_i
        c = torch.sum(torch.log(sigma_e.clamp_min(min=eps)) - torch.log(sigma_r), dim = 2)
        #print('c = ', c)
        
        score = self.gamma.item() - 0.5 * (a + b - c - self.hidden_dim)
        
        return score
       
    
    def normalize_embedding(self):
        self.entity_embedding.data.copy_(torch.renorm(input=self.entity_embedding.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        
        self.relation_embedding.data.copy_(torch.renorm(input=self.relation_embedding.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        
        self.entity_cov.data.copy_(torch.clamp(input=self.entity_cov.detach().cpu(),
                                                       min=self.cmin,
                                                       max=self.cmax))
        
        self.relation_cov.data.copy_(torch.clamp(input=self.relation_cov.detach().cpu(),
                                                       min=self.cmin,
                                                       max=self.cmax))



    
    
   
class KG2E_EL(KGEModel):
    def __init__(self, model_name, num_entity, num_relation, hidden_dim, gamma, cmin, cmax):
        super(KG2E_EL, self).__init__()
        self.name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.cmin = cmin
        self.cmax = cmax
        self.epsilon = 2.0
        self.log_2_pi = math.log(2. * math.pi)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.entity_cov = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=self.cmin,
            b=self.cmax
        )

        self.relation_cov = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=self.cmin,
            b=self.cmax
        )
        
       
    def func(self, head, rel, tail, batch_type, head_v, rel_v, tail_v):
        """
        Calculate similarity based on expected likelihood
        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
            = \frac{1}{2} \left(
                (\mu_e - \mu_r)^T(\Sigma_e + \Sigma_r)^{-1}(\mu_e - \mu_r)
                + \log \det (\Sigma_e + \Sigma_r) + d \log (2 \pi)
            \right)
            = \frac{1}{2} \left(
                \mu^T\Sigma^{-1}\mu
                + \log \det \Sigma + d \log (2 \pi)
            \right)
        """
        eps = 1.0e-10
        mu_e = head - tail
        sigma_e = head_v - tail_v
        mu_r = rel
        sigma_r = torch.clamp_min(rel_v, min=eps)
        
        #: a = \mu^T\Sigma^{-1}\mu
        """
        if batch_type == BatchType.HEAD_BATCH:
            a = torch.sum((mu_e - mu_r) ** 2 / (sigma_e + sigma_r), dim=2)
        else:
            a = torch.sum((mu_r - mu_e) ** 2 / (sigma_e + sigma_r), dim=2)
        """
        a = torch.sum((mu_e - mu_r) ** 2 / (sigma_e + sigma_r), dim=2)
        #print('a = ', a)
        
        #: b = \log \det \Sigma
        b = torch.sum(torch.log(sigma_e + sigma_r), dim=2)
        #print('b =', b)
        
        score = self.gamma.item() - 0.5 * (a + b + self.hidden_dim * self.log_2_pi)
        
        return score
       
    
    def normalize_embedding(self):
        self.entity_embedding.data.copy_(torch.renorm(input=self.entity_embedding.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        
        self.relation_embedding.data.copy_(torch.renorm(input=self.relation_embedding.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        
        self.entity_cov.data.copy_(torch.clamp(input=self.entity_cov.detach().cpu(),
                                                       min=self.cmin,
                                                       max=self.cmax))
        
        self.relation_cov.data.copy_(torch.clamp(input=self.relation_cov.detach().cpu(),
                                                       min=self.cmin,
                                                       max=self.cmax))
