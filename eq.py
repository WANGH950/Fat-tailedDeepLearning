import torch
from math import gamma

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
       
    def power_law(self, size, r_max=torch.inf):
        jump_size = self.r * torch.rand([size,1])**(-1/self.beta)
        jump_size[jump_size>r_max] = r_max
        if self.d == 1:
            return torch.sign(torch.randn([size,1]))*jump_size
        else:
            theta = torch.pi * torch.rand([size,self.d-2])
            phi = 2 * torch.pi * torch.rand([size, 1])
            angles = torch.cat([theta, phi], dim=-1)
            component = []
            for i in range(self.d):
                if i == 0:
                    component.append(jump_size * torch.cos(angles[:,i:i+1]))
                elif i == self.d - 1:
                    sin_product = torch.prod(torch.sin(angles[:,:i]), dim=1, keepdim=True)
                    component.append(jump_size * sin_product)
                else:
                    sin_product = torch.prod(torch.sin(angles[:,:i]), dim=1, keepdim=True)
                    component.append(jump_size * sin_product * torch.cos(angles[:,i:i+1]))
            return torch.cat(component, dim=1)


    def SDE(self, x, size, N, r_max=torch.inf):
        dt = self.T / N
        dB = torch.randn([size,N,self.d],device=x.device)*torch.sqrt(dt)
        jump_size = torch.zeros([size,N,self.d],device=x.device)
        Xt = torch.ones([size,N+1,self.d],device=x.device)*x
        num_jumps = torch.poisson(torch.ones([size,N],device=x.device)*self.cc*dt).int()
        for i in range(N):
            Nt = num_jumps[:,i]
            jump_size_batch = self.power_law(Nt.sum(),r_max)
            ind = torch.cumsum(torch.cat([torch.tensor([0],device=x.device),Nt]),dim=0)
            jump_size[:,i] = torch.cat([torch.sum(jump_size_batch[ind[j]:ind[j+1]],dim=0,keepdim=True) for j in range(size)])
            Xt[:,i+1] = Xt[:,i] + self.mu(Xt[:,i],i*dt)*dt + self.sigma*dB[:,i] + jump_size[:,i]
        return Xt, dB, jump_size