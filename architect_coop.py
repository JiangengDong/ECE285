import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


class Architect(object):

    def __init__(self, model, model1, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.model1 = model1
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.optimizer1 = torch.optim.Adam(self.model1.arch_parameters(),
                                           lr=args.arch_learning_rate,
                                           betas=(0.5, 0.999),
                                           weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self,
                                input,
                                target,
                                input_external,
                                target_external,
                                eta, eta1,
                                network_optimizer,
                                network_optimizer1):
        loss = self.model._loss(input, target)
        external_out = self.model(input_external)
        external_out1 = self.model1(input_external)
        softlabel_other = F.softmax(external_out1, 1)
        loss_soft = softXEnt(external_out, softlabel_other)

        # unrolling for the second model.
        loss1 = self.model1._loss(input, target)
        softlabel_other1 = F.softmax(external_out, 1)
        loss_soft1 = softXEnt(external_out1, softlabel_other1)

        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        final_loss = loss1 + loss + \
            self.args.weight_lambda * (loss_soft + loss_soft1)
        final_loss.backward()
        grad_model = [v.grad.data for v in self.model.parameters()]
        dtheta = _concat(grad_model).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))

        # for the second model.
        theta = _concat(self.model1.parameters()).data
        try:
            moment = _concat(network_optimizer1.state[v]['momentum_buffer']
                             for v in self.model1.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        grad_model1 = [v.grad.data for v in self.model1.parameters()]
        dtheta = _concat(grad_model1).data + self.network_weight_decay * theta
        unrolled_model1 = self._construct_model_from_theta1(
            theta.sub(eta1, moment + dtheta))

        return unrolled_model, unrolled_model1

    def step(self,
             input_train,
             target_train,
             input_external,
             target_external,
             input_valid,
             target_valid,
             eta,
             eta1,
             network_optimizer,
             network_optimizer1,
             unrolled):
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train,
                input_external, target_external,
                input_valid, target_valid, eta,
                eta1,
                network_optimizer, network_optimizer1)
        else:
            self._backward_step(
                input_valid,
                target_valid)
        nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.args.grad_clip)
        nn.utils.clip_grad_norm_(self.model1.arch_parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.optimizer1.step()

    def _backward_step(self,
                       input_valid,
                       target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss1 = self.model1._loss(input_valid, target_valid)
        loss = loss + loss1
        loss.backward()
        # loss1.backward()

    def _backward_step_unrolled(self,
                                input_train, target_train,
                                input_external, target_external,
                                input_valid, target_valid,
                                eta, eta1, network_optimizer,
                                network_optimizer1):
        unrolled_model, unrolled_model1 = self._compute_unrolled_model(
            input_train, target_train,
            input_external, target_external,
            eta, eta1, network_optimizer,
            network_optimizer1)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        unrolled_loss1 = unrolled_model1._loss(input_valid, target_valid)

        # loss = unrolled_loss + unrolled_loss1
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        unrolled_loss1.backward()
        dalpha1 = [v.grad for v in unrolled_model1.arch_parameters()]
        vector1 = [v.grad.data for v in unrolled_model1.parameters()]
        implicit_grads_model1_w1 = self._hessian_vector_product_model1_w1(
            vector, input_train, target_train,
            input_external, target_external)
        implicit_grads_model1_w2 = self._hessian_vector_product_model1_w2(
            vector1, input_train, target_train,
            input_external, target_external)

        implicit_grads_model2_w1 = self._hessian_vector_product_model2_w1(
            vector, input_train, target_train,
            input_external, target_external)
        implicit_grads_model2_w2 = self._hessian_vector_product_model2_w2(
            vector1, input_train, target_train,
            input_external, target_external)

        # for the second model.
        for g, ig in zip(dalpha, implicit_grads_model1_w1):
            g.data.sub_(eta, ig.data)
        for g, ig in zip(dalpha, implicit_grads_model1_w2):
            g.data.sub_(eta1, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        # for the second model.
        for g, ig in zip(dalpha1, implicit_grads_model2_w1):
            g.data.sub_(eta, ig.data)
        for g, ig in zip(dalpha1, implicit_grads_model2_w2):
            g.data.sub_(eta1, ig.data)

        for v, g in zip(self.model1.arch_parameters(), dalpha1):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta1(self, theta):
        model_new = self.model1.new()
        model_dict = self.model1.state_dict()

        params, offset = {}, 0
        for k, v in self.model1.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product_model1_w1(self,
                                          vector,
                                          input,
                                          target,
                                          input_external,
                                          target_external,
                                          r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model1(input_external), 1)
        loss_soft = softXEnt(self.model(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model(input_external), 1)
        loss_soft1 = softXEnt(self.model1(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # loss = self.model._loss(input, target)
        loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model1(input_external), 1)
        loss_soft = softXEnt(self.model(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model(input_external), 1)
        loss_soft1 = softXEnt(self.model1(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_model1_w2(self,
                                          vector,
                                          input,
                                          target,
                                          input_external,
                                          target_external,
                                          r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model1.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model1._loss(input, target)
        # loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model(input_external), 1)
        loss_soft = softXEnt(self.model1(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model1(input_external), 1)
        loss_soft1 = softXEnt(self.model(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model1.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model1._loss(input, target)
        softlabel_other = F.softmax(self.model(input_external), 1)
        loss_soft = softXEnt(self.model1(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model1(input_external), 1)
        loss_soft1 = softXEnt(self.model(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model1.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_model2_w1(self,
                                          vector,
                                          input,
                                          target,
                                          input_external,
                                          target_external,
                                          r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model1(input_external), 1)
        loss_soft = softXEnt(self.model(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model(input_external), 1)
        loss_soft1 = softXEnt(self.model1(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_p = torch.autograd.grad(loss, self.model1.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # loss = self.model._loss(input, target)
        loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model1(input_external), 1)
        loss_soft = softXEnt(self.model(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model(input_external), 1)
        loss_soft1 = softXEnt(self.model1(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_n = torch.autograd.grad(loss, self.model1.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_model2_w2(self,
                                          vector,
                                          input,
                                          target,
                                          input_external,
                                          target_external,
                                          r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model1.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model1._loss(input, target)
        # loss = self.model._loss(input, target)
        softlabel_other = F.softmax(self.model(input_external), 1)
        loss_soft = softXEnt(self.model1(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model1(input_external), 1)
        loss_soft1 = softXEnt(self.model(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_p = torch.autograd.grad(loss, self.model1.arch_parameters())

        for p, v in zip(self.model1.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model1._loss(input, target)
        softlabel_other = F.softmax(self.model(input_external), 1)
        loss_soft = softXEnt(self.model1(input_external), softlabel_other)
        softlabel_other1 = F.softmax(self.model1(input_external), 1)
        loss_soft1 = softXEnt(self.model(input_external), softlabel_other1)
        loss = loss + self.args.weight_lambda * (loss_soft + loss_soft1)

        grads_n = torch.autograd.grad(loss, self.model1.arch_parameters())

        for p, v in zip(self.model1.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
