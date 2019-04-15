import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable 
import onmt
from .loss import LossComputeBase

class AdaSoftmaxGenerator(nn.Module):
    def __init__(self, input_size, cutoff, mos=False):
        super().__init__()
        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1
        self.lang_id = []
        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()

        for i in range(len(cutoff) - 1):
            seq = nn.Sequential(
                #    nn.Linear(input_size, input_size // 4 ** i, False),
                #nn.Linear(input_size // 4 ** i, cutoff[i + 1] - cutoff[i], False)
                nn.Linear(input_size, cutoff[i + 1] - cutoff[i])
            )
            self.tail.append(seq)

        self.criterion_1 = nn.CrossEntropyLoss(reduction='sum')

    def set_target(self, target, remap_target=True):
        with torch.no_grad():
            self.id = []
            if remap_target:
                self.new_target = [target.clone()]
                for i in range(len(self.cutoff) - 1):
                    mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
                    self.new_target[0][mask] = self.cutoff[0] + i
                    if mask.sum() > 0:
                        #self.id.append(Variable(mask.float().nonzero().squeeze(1)))
                        self.id.append(mask.float().nonzero().squeeze(1))
                        self.new_target.append(target[mask].add(-self.cutoff[i]))
                    else:
                        self.id.append(None)
                        self.new_target.append(None)
            else:
                for i in range(len(self.cutoff) - 1):
                    mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
                    if mask.sum() > 0:
                        #self.id.append(Variable(mask.float().nonzero().squeeze(1)))
                        self.id.append(mask.float().nonzero().squeeze(1))
                    else:
                        self.id.append(None)

    def set_lang(self, lang_ids):
        self.lang_id = []
        with torch.autograd.no_grad():
            for i in range(len(self.cutoff) -  1):
                #mask = lang_ids.eq(self.cutoff[i])
                mask = lang_ids.eq(i+4.0)
                if mask.sum() > 0:
                    self.lang_id.append(mask.float().nonzero().squeeze(1))
                else:
                    self.lang_id.append(None)

    def AdaSoftmaxForward(self, input):
        self.output = [self.head(input)]
        for i in range(len(self.id)):
            if self.id[i] is not None:
                self.output.append(self.tail[i](input.index_select(0, self.id[i])))
            else:
                self.output.append(None)
        return None

    #return logprob
    def forward(self, input):
        with torch.autograd.no_grad():
            lsm = nn.LogSoftmax(dim=1).cuda()
            head_out = self.head(input)#.cpu()

            batch_size = head_out.size(0)
            prob = torch.zeros(batch_size, self.cutoff[-1], device=input.device, dtype=input.dtype).fill_(-1e7)
  
            lsm_head = lsm(head_out)
            prob.narrow(1, 0, self.cutoff[0]).copy_(lsm_head.narrow(1, 0, self.cutoff[0]))

            if len(self.lang_id) > 0:
                for i in range(len(self.cutoff)-1):
                    id = self.lang_id[i]
                    if id is not None:
                        pos = self.cutoff[i]
                        i_size = self.cutoff[i+1] - pos
                        buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1).index_select(0, id)
                        logit = self.tail[i](input.index_select(0, id))
                        lsm_tail = lsm(logit)
                        lsm_tail.add_(buffer)
                        prob.narrow(1, pos, i_size).index_copy_(0,id,lsm_tail)
                self.lang_id = []
            else:
                for i in range(len(self.tail)):
                    pos = self.cutoff[i]
                    i_size = self.cutoff[i + 1] - pos
                    buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1) 
                    logit = self.tail[i](input)#.cpu()
                    lsm_tail = lsm(logit)
                    prob.narrow(1, pos, i_size).copy_(buffer).add_(lsm_tail)
                    #print(prob.shape, buffer.shape, lsm_tail.shape)
            return prob

    def loss(self, criterion_0):
        output = self.output
        batch_size = output[0].size(0)
        target = self.new_target
        #criterion_1 = nn.CrossEntropyLoss(reduction='sum')
        #loss = criterion_0(output[0], Variable(target[0]))
        loss = criterion_0(output[0],target[0])

        for i in range(1, len(output), 1):
            if output[i] is not None:
                #assert(target[i].min() >= 0 and target[i].max() <= output[i].size(1))
                #loss += self.criterion_1(output[i], Variable(target[i]))
                loss += self.criterion_1(output[i],target[i])

        return loss

class NMTLossComputeBeta(LossComputeBase):
    """
    Beta NMT Loss Computation.
    """
    def __init__(self, criterion, generator, is_train=True):
        super(NMTLossComputeBeta, self).__init__(criterion, generator)
        self.generator = generator
        self.criterion = criterion
        self.is_train = is_train

    def _make_shard_state(self, batch, output, range_, attns=None):
        target = batch.tgt[range_[0] + 1: range_[1]]
        lang_id = batch.src[0][0,:,:].expand_as(target)
        return {
                "output": output,
                "target": target,
                "lang_id": lang_id,
                }

    def _compute_loss(self, batch, output, target, lang_id):
        input = output.view(-1, self.generator.input_size)
        gtruth = target.view(-1)
        self.generator.set_target(gtruth.data, remap_target=True)
        self.generator.AdaSoftmaxForward(input)

        loss = self.generator.loss(self.criterion)
        loss_data = loss.data.clone()

        if self.is_train is False:
            self.generator.set_lang(lang_id.view(-1))
            prob = self.generator(input) #None
        else:
            prob = None
        #print(loss)#, prob.exp().sum(dim=1))
        stats = self._stats(loss_data, prob, gtruth)
        return loss, stats

'''
ct = nn.NLLLoss(ignore_index=0, size_average=False)
g = AdaSoftmaxGenerator(input_size=100, cutoff=[10, 30, 60]).cuda()
c = NMTLossComputeBeta(ct, g).cuda()

output=torch.autograd.Variable(torch.randn(3, 100), requires_grad=True).cuda()
target=Variable(11*torch.ones((3, 1), dtype=torch.long)).cuda()
#Variable(torch.LongTensor([3,100])).cuda()
loss, stats = c._compute_loss(None, output=output, target=target)
loss.backward()
#print(loss.item(), output.grad, g(-output).exp(), stats)
#print('---')

#print(g.new_target)
'''
