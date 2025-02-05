import time
import eq
import utils
import torch
import torch.nn as nn
    
class BSDE(nn.Module):
    def __init__(self, N, d, T, beta, r, mu, g, f, hidden_dim=64, num_layers=3, r_max=100):
        super(BSDE, self).__init__()
        self.N = N
        self.dt = T / N
        self.r_max = r_max
        self.equation = eq.FPDE(T, beta, r, d, mu, g, f)
        self.grad = nn.ModuleList([utils.FNN(d, d, hidden_dim, num_layers) for _ in range(N)])
        self.jump = nn.ModuleList([utils.FNN(d, 1, hidden_dim, num_layers) for _ in range(N)])
        self.u = nn.Parameter(torch.rand(1), requires_grad=True)
    
    def forward(self, x, batch, sample_size):
        Xt, dBt, jump_size = self.equation.SDE(x,batch,self.N,self.r_max)
        u = torch.ones([batch,1],device=Xt.device) * self.u
        mc_js = self.equation.power_law(sample_size,self.r_max).to(Xt.device)
        for i in range(self.N):
            grad_u = self.grad[i](Xt[:,i]).unsqueeze(1)
            ui = self.jump[i](Xt[:,i])
            mc_mean = self.jump[i]((Xt[:,i].unsqueeze(1)+mc_js.unsqueeze(0)).reshape([batch*sample_size,self.equation.d])).reshape([batch,sample_size,1]).mean(dim=1)
            totel_jump = self.jump[i](Xt[:,i]+jump_size[:,i]) - ui - (mc_mean-ui)*self.equation.cc*self.dt
            u = u - self.equation.f(u)*self.dt + torch.bmm(grad_u,dBt[:,i].unsqueeze(2)).squeeze(-1) + totel_jump
        return u, self.equation.g(Xt[:,self.N])


    
class BSDETensor(nn.Module):
    def __init__(self, N, d, T, beta, r, mu, g, f, tensor_size=256, hidden_dim=64, num_layers=3, r_max=50):
        super(BSDETensor, self).__init__()
        self.N = N
        self.dt = T / N
        self.r_max = r_max
        self.equation = eq.FPDE(T, beta, r, d, mu, g, f)
        self.grad = nn.ModuleList([utils.FNN(d, d, hidden_dim, num_layers) for _ in range(N)])
        self.jumpx = nn.ModuleList([utils.FNN(d, tensor_size, tensor_size*2, num_layers) for _ in range(N)])
        self.jumpy = utils.FNN(d, tensor_size, tensor_size*2, num_layers)
        self.u = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x, batch, sample_size):
        Xt, dBt, jump_size = self.equation.SDE(x,batch,self.N,self.r_max)
        mc_js = self.equation.power_law(sample_size,self.r_max).to(x.device)
        u = torch.ones([batch,1],device=x.device) * self.u
        mc_mean = (self.jumpy(mc_js)*torch.tanh(mc_js.norm(dim=1,keepdim=True))).mean(dim=0)*self.equation.cc
        for i in range(self.N):
            grad_u = self.grad[i](Xt[:,i]).unsqueeze(1)
            totel_jump = (self.jumpx[i](Xt[:,i])*(self.jumpy(jump_size[:,i])*torch.tanh(jump_size[:,i].norm(dim=1,keepdim=True)) - mc_mean*self.dt)).sum(dim=1,keepdim=True)
            u = u - self.equation.f(u)*self.dt + torch.bmm(grad_u,dBt[:,i].unsqueeze(2)).squeeze(-1) + totel_jump
        return u, self.equation.g(Xt[:,self.N])


def train(model, params:dict):
    epoch = params['epoch']
    batch = params['batch']
    sample_size = params['sample_size']
    lr = params['lr']
    step_size = params['step_size']
    gam = params['gamma']
    x = params['x']

    optim = torch.optim.Adam(model.parameters(),lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gam)
    loss_fun = nn.MSELoss()
    
    loss_values = torch.zeros([epoch])
    res_values = torch.zeros([epoch])
    start = time.time()
    for i in range(epoch):
        model.train()
        optim.zero_grad()
        (u_pre,u_rel) = model(x,batch,sample_size)
        loss = loss_fun(u_pre,u_rel)
        loss.backward()
        optim.step()
        sched.step()

        model.eval()
        loss_values[i] = loss.item()
        res_values[i] = model.u[0].detach()
        print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e, result: %f]'.format(
            epoch,
            "#"*int((i+1)/epoch*50),
            " "*(50-int((i+1)/epoch*50)),
            time.time() - start) %
            (i+1,loss_values[i],res_values[i]), end = ' ', flush=True)
    print("\nTraining has been completed.")
    return loss_values, res_values