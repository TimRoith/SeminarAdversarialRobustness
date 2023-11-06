import torch
import torch.nn as nn

def negative_CELoss(yy,y):
    return -nn.CrossEntropyLoss()(yy,y)

class attack:
    def __init__(self, loss=None, 
                 epsilon=0.1, 
                 targeted=False,
                 verbosity = 1,
                 break_on_success = False):
        if loss is None:
            if targeted:
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = negative_CELoss
        else:
            self.loss = loss

        self.epsilon = epsilon
        self.targeted = targeted
        self.verbosity = verbosity
        self.break_on_success = break_on_success

    def __call__(self, model, x, y = None):
        if y is None:
            y = model(x).argmax(dim=-1)
        
        return self.forward(model, x, y)
    
    def forward(self, model, x, y):
        raise NotImplementedError()
    
class pgd(attack):
    def __init__(self,
                 x_min=0.0, x_max=1.0, 
                 restarts=1, 
                 attack_iters=7, 
                 alpha=0.1, alpha_mul=1.0,
                 init_mode = 'uniform',
                 mask = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.x_min = x_min
        self.x_max = x_max
        self.restarts = restarts
        self.attack_iters = attack_iters
        self.alpha = alpha
        self.alpha_mul = alpha_mul
        self.init_mode = init_mode

        if mask is None:
            self.mask = 1.
        else:
            self.mask = mask

    def update_idx(self, yy, y):
        if self.break_on_success:
            if self.targeted:
                ind = torch.where(yy != y)[0]
            else:
                ind = torch.where(yy == y)[0]

            if len(ind) == 0:
                self.index = None
            else:
                self.index = ind
                
        else: 
            self.index = Ellipsis

    def forward(self, model, x, y):
        for i in range(self.restarts): 
            self.init_delta(x)
            self.it = 0

            while self.it < self.attack_iters:
                inp = torch.clamp(x + self.mask * self.delta, self.x_min, self.x_max)
                pred = model(inp)
                
                # indexes are used to determine which samples needs to be updated
                self.update_idx(pred.max(1)[1], y)
                if self.index is None:
                    break

                # get loss and step backward
                loss = self.loss(pred, y)
                loss.backward()

                # perform inner step
                self.inner_update()

                # update alpha
                self.alpha *= self.alpha_mul
                self.it += 1
                self.print_update(loss.item())

        return torch.clamp(x + self.delta.detach(), self.x_min, self.x_max)



    def print_update(self, loss):
        if self.verbosity > 0:
            print('Iteration ' + str(self.it) + 
                ', loss: ' + str(loss))
    
    def init_delta(self, x):
        d = torch.zeros_like(x)
        if self.init_mode == 'uniform':
            d.uniform_(-self.epsilon, self.epsilon)
            d = clamp(d, self.x_min - x, self.x_max - x)
        self.delta =  self.project(d, r=self.epsilon)
        self.delta.requires_grad = True
    
    def inner_update(self,):
        grad = self.delta.grad.detach()
        d = self.delta - self.alpha * grad
        d = self.project(d, self.epsilon)
        self.delta.data[self.index] = d[self.index]
       
        self.delta.grad.zero_()
        self.delta.data *= self.mask
        
    def project(self, d, r=1):
        raise NotImplementedError
    
class pgsd(pgd):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def inner_update(self,):
        grad = self.delta.grad.detach()
        d = self.delta - self.alpha * torch.sign(grad)
        d = self.project(d, self.epsilon)
        self.delta.data[self.index] = d[self.index]
       
        self.delta.grad.zero_()
        self.delta.data *= self.mask

class L2_pgd(pgd):
    def __init__(self, epsilon=2.0, alpha=0.5, **kwargs):
        super().__init__(epsilon=epsilon, alpha=alpha, **kwargs)
    
    def project(self, d, r=1):
        return d / torch.norm(d.view(d.shape[0], -1), p=2, dim=1).view(d.shape[0], 1, 1, 1) * r

class L2_pgsd(pgsd):
    def __init__(self, epsilon=2.0, alpha=0.5, **kwargs):
        super().__init__(epsilon=epsilon, alpha=alpha, **kwargs)
    
    def project(self, d, r=1):
        return d / torch.norm(d.view(d.shape[0], -1), p=2, dim=1).view(d.shape[0], 1, 1, 1) * self.epsilon

class Linf_pgd(pgd):
    def __init__(self, epsilon=1.0, alpha=0.3/4, **kwargs):
        super().__init__(epsilon=epsilon, alpha=alpha, **kwargs)

    def project(self, d, r=1):
        return torch.clamp(d, -r, r)
    
class Linf_pgsd(pgsd):
    def __init__(self, epsilon=1.0, alpha=0.3/4, **kwargs):
        super().__init__(epsilon=epsilon, alpha=alpha, **kwargs)

    def project(self, d, r=1):
        return torch.clamp(d, -r, r)

                                     
def clamp(x, x_min, x_max):
    return torch.max(torch.min(x, x_max), x_min)
                    
def get_delta(x, eps=1.0, uniform=False, x_min=0.0, x_max=1.0):
    delta = torch.zeros_like(x)
    if uniform:
        delta.uniform_(-eps, eps)
    return clamp(delta, x_min - x, x_max - x)