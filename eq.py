import torch
from math import gamma

default_device = torch.get_default_device()

class FPDE():
    def __init__(self, T, beta, r, d, mu, g, f):
        self.T = torch.tensor(T)
        self.beta = torch.tensor(beta)
        self.r = torch.tensor(r)
        self.d = torch.tensor(d)
        self.mu = mu
        self.g = g
        self.f = f
        M = int(d/2)-1
        if M <= 0:
            self.cc = 2**(beta+1)/torch.abs(torch.tensor(gamma(-beta/2)))/beta/r**beta*gamma((beta+d)/2)/gamma(d/2)
            self.sigma = torch.sqrt(2**beta*r**(2-beta)*gamma((d+beta)/2)/gamma(d/2+1)/torch.abs(torch.tensor(gamma(-beta/2)))/(2-beta))
        else:
            self.cc = 2**(beta+1)/torch.abs(torch.tensor(gamma(-beta/2)))/beta/r**beta*gamma((beta+d)/2-M)/gamma(d/2-M)
            self.sigma = torch.sqrt(2**beta*r**(2-beta)*gamma((d+beta)/2-M)/gamma(d/2+1-M)/torch.abs(torch.tensor(gamma(-beta/2)))/(2-beta))
            for k in range(M):
                self.cc *= (d+beta-2*(k+1))/(d-2*(k+1))
                self.sigma *= torch.sqrt(torch.tensor((d+beta-2*(k+1))/(d+2-2*(k+1))))
       
    def power_law(self, size, r_max=torch.inf, device=default_device):
        rand = torch.rand([size,1],device=device)
        jump_size = self.r * rand**(-1/self.beta)
        jump_size[rand<(self.r/r_max)**self.beta] = r_max
        if self.d == 1:
            return torch.sign(torch.randn([size,1],device=device))*jump_size
        else:
            theta = torch.pi * torch.rand([size,self.d-2],device=device)
            phi = 2 * torch.pi * torch.rand([size, 1],device=device)
            angles = torch.cat([theta, phi], dim=-1)
            sin_product = torch.cat([torch.ones([size,1],device=device),torch.cumprod(torch.sin(angles), dim=1)],dim=1)
            cos_angles = torch.cat([torch.cos(angles),torch.ones([size,1],device=device)],dim=1)
            return jump_size * sin_product * cos_angles


    def SDE(self, x, size, N, r_max=torch.inf, device=default_device):
        dt = self.T / N
        dB = torch.randn([N,size,self.d],device=device)*torch.sqrt(dt)
        jump_size = torch.zeros([N,size,self.d],device=device)
        Xt = torch.ones([N+1,size,self.d],device=device)*x
        num_jumps = torch.poisson(torch.ones([N,size],device=device)*self.cc*dt).int()
        for i in range(N):
            jump_size_batch = self.power_law(num_jumps[i].sum(),r_max,device)
            jump_ornot = num_jumps[i]>0
            jump = [torch.zeros([0,self.d],device=device)]
            jump_num = 0
            for j in range(jump_ornot.sum()):
                jump.append(torch.sum(jump_size_batch[jump_num:jump_num+num_jumps[i,jump_ornot][j]],dim=0,keepdim=True))
                jump_num += num_jumps[i,jump_ornot][j]
            jump_size[i,jump_ornot] = torch.cat(jump)
            Xt[i+1] = Xt[i] + self.mu(Xt[i],i*dt)*dt + self.sigma*dB[i] + jump_size[i]
        return Xt, dB, jump_size
    
class FPDEG():
    def __init__(self, T, lamb, d, mu, g, f, jump):
        self.T = torch.tensor(T)
        self.d = torch.tensor(d)
        self.lamb = lamb
        self.mu = mu
        self.g = g
        self.f = f
        self.jump = jump

    def SDE(self, x, size, N, device=default_device):
        dt = self.T / N
        jump_size = torch.zeros([N,size,self.d],device=device)
        Xt = torch.ones([N+1,size,self.d],device=device)*x
        num_jumps = torch.poisson(torch.ones([N,size],device=device)*self.lamb*dt).int()
        for i in range(N):
            jump_size_batch = self.jump(num_jumps[i].sum(),device)
            jump_ornot = num_jumps[i]>0
            jump = [torch.zeros([0,self.d],device=device)]
            jump_num = 0
            for j in range(jump_ornot.sum()):
                jump.append(torch.sum(jump_size_batch[jump_num:jump_num+num_jumps[i,jump_ornot][j]],dim=0,keepdim=True))
                jump_num += num_jumps[i,jump_ornot][j]
            jump_size[i,jump_ornot] = torch.cat(jump)
            Xt[i+1] = Xt[i] + self.mu(Xt[i],i*dt)*dt + jump_size[i]
        return Xt, jump_size