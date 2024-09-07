import torch


class DDPM():
    def __init__(
            self, 
            device,
            n_steps,
            min_beta=1e-4,
            max_beta=2e-2
    ):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.n_steps = n_steps
    
    def sample_forward(self, x, t, eps=None):
        alpha_cumprod = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_cumprod) + torch.sqrt(alpha_cumprod) * x
        return res
    
    def sample_backward(self, img_shape, net, device, simple_var=True):
        """
        随机生成纯噪声`x`，再令`t`从`n_steps-1`到0，调用`sample_forward_step()`，逐步生成样本
        """
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x
    
    def sample_backward_step(self, x_t, t, net, simple_var=True):
        """

        """
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1-self.alphas_cumprod[t-1])/(1-self.alphas_cumprod[t])*self.betas[t]
            
            noise = torch.randn_like(x_t) * torch.sqrt(var)

        mean = (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*eps)/torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t