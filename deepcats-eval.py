import torch
import zipfile
import random
import os
import math
import adabelief_pytorch
from PIL import Image
from torchvision import tv_tensors
from torchvision.utils import save_image
from functorch.compile import memory_efficient_fusion, aot_module,min_cut_rematerialization_partition,ts_compile


cuda = torch.device('cuda')
torch.set_default_device(cuda)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.autograd.grad_mode.set_grad_enabled(False)



class Arctan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.atan()
  

        
class Mean(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.mean()
mean_mod = Mean()
class Sum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum()
sum_mod = Sum()

        
class Printsize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print(input.size())
        return input
class Transpose(torch.nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y

    def forward(self, input):
        return input.transpose(self.x,self.y)
class ConstantMul(torch.nn.Module):
    def __init__(self,x):
        super().__init__()
        self.x = x

    def forward(self, input):
        return input.mul(self.x)

class FlipAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        split = input.size(0) // 2
        return torch.cat([input[:split],torch.flip(input[split:],[-1])],0)
class FlipAugment1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cat([input,torch.flip(input,[-1])],0)
class ConstantDiv(torch.nn.Module):
    def __init__(self,x):
        super().__init__()
        self.x = x

    def forward(self, input):
        return input.div(self.x)

class Negate(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.neg()




class ArLUv2(torch.nn.Module):
    def __init__(self,dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.donorm = math.sqrt(1.0 - dropout)

    def forward(self, input):
        do = self.dropout if self.training else 0.0
        
        if do > 0.0:
            y = torch.rand((input.size()[:-2] + [1,1]),dtype=input.dtype,device=input.device).bernoulli_(1.0 - do)
            input = input.mul(y)
            y = None
            
            
        x = input.atan()
        
        x = torch.cat([x.maximum(input),x.minimum(input)],-3)
        if do > 0.0:
            return x.div(self.donorm)
        return x

class ArLUv2_flat(torch.nn.Module):
    def __init__(self,dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.donorm = math.sqrt(1.0 - dropout)

    def forward(self, input):
        do = self.dropout if self.training else 0.0
        
        if do > 0.0:
            y = torch.rand_like(input).bernoulli_(1.0 - do)
            input = input.mul(y)
            y = None
            
            
        x = input.atan()
        
        x = torch.cat([x.maximum(input),x.minimum(input)],-1)
        if do > 0.0:
            return x.div(self.donorm)
        return x
        





arctan_mod = Arctan()




def makekaiminglinear(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain / math.sqrt(inputs))
    return lin
def makekaiminglinear2(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain / math.sqrt(outputs))
    return lin

def makemanuallinear(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain)
    return lin

def makemanuallinearuniform(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.uniform_(lin.weight,-gain,gain)
    return lin

def makezerolinear(inputs, outputs, bias = True):
    lin = torch.nn.Linear(inputs, outputs, bias)
    with torch.no_grad():
        lin.weight.zero_()
    return lin

sqrtgain = math.sqrt(2)


def biasinit(layer,gain=0.0):
    with torch.no_grad():
        layer.bias.fill_(gain)
    return layer


def convinitnb(layer,gain=1.0):
    myw = layer.weight
    mysize = myw.size()
    torch.nn.init.normal_(myw,0.0,gain / math.sqrt(mysize[0] * mysize[2] * mysize[3]))
    return layer
def convinitmanual(layer,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain)
    return layer
def convinitmanualuniform(layer,gain=1.0):
    torch.nn.init.uniform_(layer.weight,-gain,gain)
    return layer

def convinit2nb(layer,m=4,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain / math.sqrt(layer.weight.size()[0] * m))
    return layer.to(memory_format=torch.channels_last)

def convinit3nb(layer,m=4,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain / math.sqrt(layer.weight.size()[1] * m))
    return layer.to(memory_format=torch.channels_last)






batchSize = 128


sqrt2 = math.sqrt(2.0)

#onesmask = torch.ones(batchSize,1)







        
class SelfAttentionThing(torch.nn.Module):
    def __init__(self,size,gain):
        super().__init__()
        gain /= math.sqrt(size)
        self.keys = torch.nn.Parameter(torch.randn(size,size).mul_(gain))
        self.queries = torch.nn.Parameter(torch.randn(size,size).mul_(gain))

    def forward(self, input):
        x = input.transpose(-1,-3)
        x = x.flatten(-3,-2)
        x = x.unsqueeze(-3)
        x = torch.nn.functional.scaled_dot_product_attention(x.matmul(self.queries), x.matmul(self.keys),x)
        x = x.squeeze(-3)
        x = x.unflatten(-2,input.size()[-2:])
        x = x.transpose(-1,-3)
        return torch.cat([input,x],-3).to(memory_format=torch.channels_last)

class AugmentKernel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        p = torch.nn.functional.pad(input,(1,1,1,1),'replicate')
        return torch.cat([p,torch.nn.functional.avg_pool2d(p,3,padding=1,stride=1),
        torch.nn.functional.max_pool2d(torch.nn.functional.pad(torch.cat([input,input.neg()],-3), (2,2,2,2), value=(-math.inf)),3,stride=1)],-3)


class VariancePreservingDropout(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.dropout = 1.0 - dropout
        self.donorm = math.sqrt(1.0 - dropout)

    def forward(self, input):
        if self.training:
            return input
        return input.mul(torch.rand_like(input).bernoulli_(self.dropout).div_(self.donorm))

arluv2_mod = ArLUv2(0.0)
arluv2_flat_mod = ArLUv2_flat(0.0)
arluv2_flat_mod_dropout = torch.jit.script(ArLUv2_flat(0.25))

sqrt12 = math.sqrt(12)





fdinitgain = 1.0 / 5.0


#l2_regularization = 1e-3
l2_regularization_fast_discriminator = 1e-2
#l1_regularization = 1e-4


#generator_initstd = 0.02540494454



generator_gain = 1.48663410298
attn_query_gain = 1.0 / 0.90747

generator = torch.nn.Sequential(
    ConstantMul(0.90747),
    makekaiminglinear(4096,4096),arctan_mod,
    makekaiminglinear(4096,4096),arctan_mod,
    makekaiminglinear(4096,16384),arctan_mod,
    torch.nn.Unflatten(-1, (4,4,1024)),Transpose(-3,-1),Transpose(-2,-1),
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(1024,512,4,padding=1,stride=2)),4,gain=generator_gain),arctan_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(512,256,4,padding=1,stride=2)),4,gain=generator_gain),arctan_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(256,128,4,padding=1,stride=2)),4,gain=generator_gain),arctan_mod,
    convinit3nb(biasinit(torch.nn.Conv2d(128,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(128,3,4,padding=1,stride=2)),4,gain=generator_gain),
    ConstantDiv(0.90747)
)
generator.load_state_dict(torch.load("models/generator_100000", weights_only=True))

imgss = generator.forward(torch.randn(256,4096)).div_(sqrt12).add_(0.5)

imgss = torch.nn.functional.pad(imgss,(4,4,4,4))

#[c,z,x2,y2]
imgss = imgss.transpose(0,1)

#[c,y1,x1,x2,y2]
imgss = imgss.unflatten(1,(16,16))


#[c,y1,x,y2]
imgss = imgss.flatten(2,3)

imgss = imgss.transpose(1,2)
#[c,x,y1,y2]

imgss = imgss.flatten(2,3)
#[c,x,y]

save_image(imgss, "fakecats4.png")