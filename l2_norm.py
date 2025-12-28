from lib import *
class L2Norm(nn.Module):
    def __init__(self, input_channels = 512, scale=20):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10
    
    def reset_parameter(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x*weights