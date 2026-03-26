import torch
import zipfile
import random
import os
import math
import adabelief_pytorch
from PIL import Image
from torchvision import tv_tensors
from torchvision.utils import save_image
from functorch.compile import memory_efficient_fusion, aot_module,min_cut_rematerialization_partition,ts_compile, default_decompositions
from functorch.compile import nop as no_compiler
from itertools import chain
#from torchvision.transforms import Resize,InterpolationMode
#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#from torchvision.transforms.functional import resize

cuda = torch.device('cuda')
torch.set_default_device(cuda)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Arctan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.atan()
arctan_mod = Arctan()

class ERF(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.erf()  


        
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
        return torch.cat([input[:split],torch.flip(input[split:],[-1])],0).to(memory_format=torch.channels_last)
class FlipAugment1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        split = input.size(0) // 4
        return torch.cat([input[:split],torch.flip(input[split:-split],[-1]),input[-split:]],0).to(memory_format=torch.channels_last)

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


#HACK: AVOID problems with AOTAutograd
class SoftplusV2_ManGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        i = torch.cat([i,i.neg()],1)
        i = i.to(memory_format=torch.channels_last)
        return torch.nn.functional.softplus(i)
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        hs = grad_output.size(1) // 2
        i = i.sigmoid()
        gi = grad_output[:,0:hs].to(memory_format=torch.channels_last).mul(i)
        grad_output = grad_output[:,hs:]
        i = i.sub(1.0)
        gi = gi.addcmul(grad_output, i)  
        return gi

class SoftplusV2_ManGrad_Flat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        i = torch.cat([i,i.neg()],1)
        return torch.nn.functional.softplus(i)
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        hs = grad_output.size(1) // 2
        i = i.sigmoid()
        gi = grad_output[:,0:hs].contiguous().mul(i)
        grad_output = grad_output[:,hs:]
        i = i.sub(1.0)
        gi = gi.addcmul(grad_output, i)  
        return gi

class SoftplusV2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return SoftplusV2_ManGrad.apply(input)

class SoftplusV2_flat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return SoftplusV2_ManGrad_Flat.apply(input)

class AvgUnpoolMangrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        s0 = i.size(0)
        s1 = i.size(1)
        s2 = i.size(2)
        s3 = i.size(3)
        i = i.unsqueeze(-2)
        i = i.unsqueeze(-1)
        i = i.expand(s0,s1,s2,2,s3,2)
        i = i.flatten(-4,-3)
        i = i.flatten(-2,-1)
        return i.to(memory_format=torch.channels_last)
    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.avg_pool2d(grad_output,2,divisor_override=1)
class AvgUnpool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return AvgUnpoolMangrad.apply(input)


class ForceDropout2D(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.dropout = 1.0 - dropout

    def forward(self, input):
        return input.mul(torch.rand(list(input.size()[:-2]) + [1,1],dtype=input.dtype,device=input.device).bernoulli_(self.dropout))


class ForceDropout(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.dropout = 1.0 - dropout

    def forward(self, input):
        return input.mul(torch.rand_like(input).bernoulli_(self.dropout))



#AOT Autograd friendly random flipping
class BRFlip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        rze = torch.rand([input.size(0),1,1,1],dtype=input.dtype,device=input.device).bernoulli_(0.5)
        return input.mul(rze).addcmul(torch.flip(input,[-1]).to(memory_format=torch.channels_last),1.0 - rze)


brfmod = BRFlip()


softplus_mod = torch.nn.Softplus()
fdo_mod = torch.jit.script(ForceDropout2D(0.125))
fdo_mod_1 = torch.jit.script(ForceDropout(0.125))



def makekaiminglinear(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain / math.sqrt(inputs))
    return lin

def makemanuallinear(inputs, outputs, bias, gain):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain)
    return lin

def makekaiminglinear2(inputs, outputs, bias = True, gain = 1.0):
    lin = torch.nn.Linear(inputs, outputs, bias)
    torch.nn.init.normal_(lin.weight,0.0,gain / math.sqrt(outputs))
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
    





def convinit2nb(layer,m=4,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain / math.sqrt(layer.weight.size(0) * m))
    return layer.to(memory_format=torch.channels_last)



def convinit3nb(layer,m=4,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain / math.sqrt(layer.weight.size(1) * m))
    return layer.to(memory_format=torch.channels_last)

def convinitcenter(layer,off):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.normal_(layer.weight[:,:,off,off],0.0,1.0 / math.sqrt(layer.weight.size(1)))
    
    return layer.to(memory_format=torch.channels_last)

def convinit3man(layer,gain=1.0):
    torch.nn.init.normal_(layer.weight,0.0,gain)
    return layer.to(memory_format=torch.channels_last)




batchSize = 128


sqrt2 = math.sqrt(2.0)

#onesmask = torch.ones(batchSize,1)



def biasinit1(layer):
    with torch.no_grad():
        layer.bias.normal_(0.0,sqrt2)
    return layer



class BiasLayer(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input):
        return input.add(self.bias)

class BiasLayer2(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.mbias = torch.nn.Parameter(torch.ones(size))
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input):
        return self.bias.addcmul(self.mbias, input)
            
class BiasLayerCL(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size).transpose(-3,-1).transpose(-2,-1))

    def forward(self, input):
        return input.add(self.bias)




class NormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.div(input.mul(input).mean((-1),keepdim=True).sqrt())

class HalfNormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.sub(input.mean((-1),keepdim=True))

class HalfNormLayer2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.sub(input.mean((-1,-2),keepdim=True))


class NormLayer2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.div(input.mul(input).mean((-1,-2,-3),keepdim=True).sqrt())
        


norm_layer = NormLayer2d()
norm_layer_1 = NormLayer()
half_norm = HalfNormLayer()
half_norm_2 = HalfNormLayer2()


class InstanceNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.add(torch.randn_like(input),alpha=0.125)

class ForkAdd(torch.nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, input):
        return self.a.forward(input).add(self.b.forward(input))

class GatedActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        s = input.size(-3) // 2
        x = input[:,:s]
        y = input[:,s:]
        input = None
        #x = x.to(memory_format=torch.channels_last)
        x = x.div(sqrt2)
        x = x.erf()
        x = x.add(1.0)
        #y = y.to(memory_format=torch.channels_last)
        y = y.atan()
        x = x.mul(y)
        y = None
        return x


class GatedActivation1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        s = input.size(-1) // 2
        x = input[:,:s]
        y = input[:,s:]
        input = None
        #x = x.contiguous()
        x = x.div(sqrt2)
        x = x.erf()
        x = x.add(1.0)
        #y = y.contiguous()
        y = y.atan()
        x = x.mul(y)
        y = None
        return x

def a2(x):
    return x.to(memory_format=torch.channels_last) if (x.dim() == 4) else x.contiguous()

#HACK: AVOID problems with AOTAutograd
class CReLU_ManGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        i = torch.cat([i,i.neg()],1)
        i = a2(i)
        i.relu_()
        return i
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        hs = grad_output.size(1) // 2
        i = i.sign()
        i.relu_()
        gi = grad_output[:,0:hs].mul(i)
        grad_output = grad_output[:,hs:]
        i = i.sub(1.0)
        gi = gi.addcmul(grad_output, i)  
        return gi



class CReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return CReLU_ManGrad.apply(input)
crelu_mod = CReLU()





class DAA(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.d = d
    def forward(self, z):
        accu = None
        for x in self.d:
            if type(x) is torch.nn.Linear:
                szm1 = z.size(-1) // 2
                p = x.forward(torch.cat([z[:,:,:,:szm1].sum(0).add(torch.flip(z[:,:,:,szm1:].sum(0), [-1])).flatten(-3,-1), z.mul(z).sum((-1, -2)).add(1e-8).sqrt().sum(0)], 0))
                if accu is None:
                    accu = p
                else:
                    accu = accu.add(p)
                p = None
            else:
                z = x.forward(z)
        z = None
        accu = torch.squeeze(accu)
        return accu

class DAA2(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.d = d
    def forward(self, z):
        accu = None
        for x in self.d:
            if type(x) is torch.nn.Linear:
                szm1 = z.size(-1) // 2
                p = x.forward(z[:,:,:,:szm1].sum(0).add(torch.flip(z[:,:,:,szm1:].sum(0), [-1])).flatten(-3,-1))
                if accu is None:
                    accu = p
                else:
                    accu = accu.add(p)
                p = None
            else:
                z = x.forward(z)
        z = None
        accu = torch.squeeze(accu)
        return accu



def optimizer_to_cpu(optim):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.cpu()
def optimizer_to_cuda(optim):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(cuda)

class GradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return i
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.div(grad_output.mul(grad_output).sum((1,2,3), keepdim=True).sqrt().mul(grad_output.size(0)))
class GradNormMod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return GradNorm.apply(input)

class FixNanGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return i
    @staticmethod
    def backward(ctx, grad_output):
        return torch.nan_to_num(grad_output)


aup_mod = AvgUnpool()

ga = GatedActivation()


sqrt12 = math.sqrt(12)


        
softplusv2_mod = SoftplusV2()
softplusv2_flat_mod = SoftplusV2_flat()


#fg = 1.0 / math.sqrt(3)
ivs2 = 1.0 / sqrt2









erf_mod = ERF()

apm = torch.nn.AvgPool2d(2,divisor_override=2)


discriminator_softplus = DAA(torch.nn.ModuleList([
    #InstanceNoise(),
    convinit3nb(biasinit(torch.nn.Conv2d(3,128,3,padding=1,padding_mode="replicate")),9),brfmod,softplusv2_mod,makezerolinear(256*(32*64+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9,sqrt2),brfmod,softplusv2_mod,makezerolinear(256*(32*64+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,4,stride=2,padding=1,padding_mode="replicate")),16,sqrt2),brfmod,softplusv2_mod,makezerolinear(256*(16*32+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9,sqrt2),brfmod,softplusv2_mod,makezerolinear(256*(16*32+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9,sqrt2),brfmod,softplusv2_mod,makezerolinear(256*(16*32+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,256,4,stride=2,padding=1,padding_mode="replicate")),16,sqrt2),brfmod,softplusv2_mod,makezerolinear(512*(8*16+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(512,256,3,padding=1,padding_mode="replicate")),9,sqrt2),brfmod,softplusv2_mod,makezerolinear(512*(8*16+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(512,512,4,padding=1,stride=2,padding_mode="replicate")),16,sqrt2),brfmod,softplusv2_mod,makezerolinear(1024*(4*8+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(1024,512,3,padding=1,padding_mode="replicate")),9,sqrt2),brfmod,softplusv2_mod,makezerolinear(1024*(4*8+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(1024,1024,4,padding=1,stride=2,padding_mode="replicate")),16,sqrt2),brfmod,softplusv2_mod,makezerolinear(2048*(2*4+1),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(2048,1024,3,padding=1,padding_mode="replicate")),9,sqrt2),softplusv2_mod,makemanuallinear(2048*(2*4+1),1,False,1.0/math.sqrt(4096*4*4))
]))

discriminator_relu = DAA2(torch.nn.ModuleList([
    #InstanceNoise(),
    convinit3nb(biasinit(torch.nn.Conv2d(3,128,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(256*(32*64),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(256*(32*64),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,4,stride=2,padding=1,padding_mode="replicate")),16),brfmod,crelu_mod,makezerolinear(256*(16*32),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(256*(16*32),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(256*(16*32),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(256,256,4,stride=2,padding=1,padding_mode="replicate")),16),brfmod,crelu_mod,makezerolinear(512*(8*16),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(512,256,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(512*(8*16),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(512,512,4,padding=1,stride=2,padding_mode="replicate")),16),brfmod,crelu_mod,makezerolinear(1024*(4*8),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(1024,512,3,padding=1,padding_mode="replicate")),9),brfmod,crelu_mod,makezerolinear(1024*(4*8),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(1024,1024,4,padding=1,stride=2,padding_mode="replicate")),16),brfmod,crelu_mod,makezerolinear(2048*(2*4),1,False),
    convinit3nb(biasinit(torch.nn.Conv2d(2048,1024,3,padding=1,padding_mode="replicate")),9),crelu_mod,makemanuallinear(2048*(2*4),1,False,1.0/math.sqrt(2048*4*4))
]))
discriminator = ForkAdd(discriminator_softplus, discriminator_relu)

def rnrd():
    for x in discriminator_relu.parameters():
        if(x.dim() == 4):
            x.div_(x.mul(x).sum((1,2,3),keepdim=True).sqrt_())
with torch.no_grad():
    rnrd()


fdo_mg = ConstantDiv(1.0 - 0.125)

px = torch.nn.PixelShuffle(2)

iv2 = 1.0 / sqrt2





ga1 = GatedActivation1()



generator = torch.nn.Sequential(
    makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    #makekaiminglinear(4096,16384*2),
    makekaiminglinear(2048,6*6*1024*2),ga1,
    #SMART TRANSPOSE because we are channels-last and we want to avoid copying
    torch.nn.Unflatten(-1, (6,6,1024)),Transpose(-3,-1),Transpose(-2,-1),fdo_mod,norm_layer,brfmod, #6x6
    convinit3nb(biasinit(torch.nn.Conv2d(1024,2048*2,3)),9),px,ga,fdo_mod,norm_layer,brfmod, #8x8
    convinit3nb(biasinit(torch.nn.Conv2d(512,512*2,3,padding=1,padding_mode="replicate")),9),ga,fdo_mod,norm_layer,brfmod,
    convinit3nb(biasinit(torch.nn.Conv2d(512,1024*2,3,padding=1,padding_mode="replicate")),9),px,ga,fdo_mod,norm_layer,brfmod, #16x16
    convinit3nb(biasinit(torch.nn.Conv2d(256,256*2,3,padding=1,padding_mode="replicate")),9),ga,fdo_mod,norm_layer,brfmod,
    convinit3nb(biasinit(torch.nn.Conv2d(256,512*2,3,padding=1,padding_mode="replicate")),9),px,ga,fdo_mod,norm_layer,brfmod, #32x32
    convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,3,padding=1,padding_mode="replicate")),9),ga,fdo_mod,norm_layer,brfmod,
    convinit3nb(biasinit(torch.nn.Conv2d(128,512*2,3,padding=1,padding_mode="replicate")),9),px,ga,fdo_mod,norm_layer,brfmod, #64x64
    convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,3,padding=1,padding_mode="replicate")),9),ga,fdo_mod,norm_layer,brfmod,
    convinit3nb(biasinit(torch.nn.Conv2d(128,3,3,padding=1,padding_mode="replicate")),9)
)

# generator = torch.nn.Sequential(
    # makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    # makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    # makekaiminglinear(2048,4096),ga1,fdo_mod_1,norm_layer_1,
    # #makekaiminglinear(4096,16384*2),
    # makekaiminglinear(2048,4*4*512*2),ga1,
    # #SMART TRANSPOSE because we are channels-last and we want to avoid copying
    # torch.nn.Unflatten(-1, (4,4,512)),Transpose(-3,-1),Transpose(-2,-1),fdo_mod,norm_layer,brfmod,aup_mod, #8x8
    # convinit3nb(biasinit(torch.nn.Conv2d(512,512*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(512,512*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(512,256*2,3)),9),ga,fdo_mod,norm_layer,brfmod,aup_mod, #16x16
    # convinit3nb(biasinit(torch.nn.Conv2d(256,256*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(256,256*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(256,128*2,3)),9),ga,fdo_mod,norm_layer,brfmod,aup_mod, #32x32
    # convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,3)),9),ga,fdo_mod,norm_layer,brfmod,aup_mod, #64x64
    # convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(128,128*2,2,padding=1,padding_mode="replicate")),4),ga,fdo_mod,norm_layer,brfmod,
    # convinit3nb(biasinit(torch.nn.Conv2d(128,3,3)),9)
# )







discriminator.train(True)
generator.train(False)
refrand = torch.empty(batchSize,2048)
generator_trace = torch.jit.trace(generator,refrand,check_trace=False)


fa_mod = FlipAugment()

discriminator_trace_softplus = memory_efficient_fusion(discriminator_softplus)
discriminator_trace_relu = memory_efficient_fusion(discriminator_relu)
#discriminator_trace_1 = memory_efficient_fusion(torch.nn.Sequential(fa_mod,discriminator))

generator.train(True)
discriminator.train(False)

#generator_trace_1 = memory_efficient_fusion(torch.nn.Sequential(generator,fa_mod))

#discriminator_trace_1 = torch.jit.trace(discriminator,static_datatape,check_trace=False)
generator_discriminator = memory_efficient_fusion(torch.nn.Sequential(generator,ForkAdd(torch.nn.Sequential(fa_mod, discriminator_softplus), torch.nn.Sequential(FlipAugment1(), discriminator_relu)),ConstantDiv(batchSize)))

discriminator.train(True)
generator.train(False)






filelist = []


for currentpath, folders, files in os.walk("."):
    for file in files:
        if file.endswith(".jpg"):
            filelist.append(os.path.join(currentpath, file))

filecount = len(filelist) - 1
        


#generator_optimizer = torch.optim.RMSprop(generator.parameters(),1e-6,0.999,eps=1e-9)
generator_optimizer = torch.optim.Adam(generator.parameters(),lr=1e-6,betas=(0.9,0.999),eps=1e-9)
#generator_optimizer = torch.optim.Adamax(generator.parameters(),lr=1e-5,eps=1e-9)
#generator_optimizer = adabelief_pytorch.AdaBelief(generator.parameters(),lr=1e-5,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)

optimizer_to_cpu(generator_optimizer)
#generator_optimizer = torch.optim.SGD(discriminator.parameters(),lr=1e-3,momentum=0.9)
discriminator_optimizer = adabelief_pytorch.AdaBelief(discriminator.parameters(),lr=2e-5,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)
#discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=1e-5,eps=1e-9)
#discriminator_optimizer = torch.optim.Adamax(discriminator.parameters(),lr=1e-4,eps=1e-9)
#discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(),lr=1e-5,eps=1e-9,alpha=0.999)
optimizer_to_cpu(discriminator_optimizer)



#generator.load_state_dict(torch.load("models/generator_50000", weights_only=True))
#generator_optimizer.load_state_dict(torch.load("models/generator_optimizer_50000", weights_only=True))
#discriminator.load_state_dict(torch.load("models/discriminator_50000", weights_only=True))
#discriminator_optimizer.load_state_dict(torch.load("models/discriminator_optimizer_50000", weights_only=True))

gpw = (64*64*3)




for x in generator.parameters():
    x.requires_grad_(True)



def dumpgrad(mod):
    with torch.no_grad():
        gsd = mod.state_dict(keep_vars=True)
        for x in mod.state_dict(keep_vars=True):
            mygrad = gsd[x].grad
            if mygrad is None:
                continue
            print(x + ": " + str(mygrad.mul(mygrad).mean().sqrt().tolist()))
            mygrad = None

ic = {}



def crop_center_square(image: Image.Image) -> Image.Image:
    """
    Crop the center square from a non-square PIL image.
    If the image is already square, it is returned unchanged.

    :param image: A PIL.Image.Image object
    :return: A square PIL.Image.Image object
    """
    width, height = image.size
    
    if width == height:
        return image  # Already square, return as is

    # Determine the size of the square (the smaller of width or height)
    new_edge = min(width, height)

    # Calculate cropping box (left, upper, right, lower)
    left = (width - new_edge) // 2
    upper = (height - new_edge) // 2
    right = left + new_edge
    lower = upper + new_edge

    # Crop and return
    return image.crop((left, upper, right, lower))
static_datatape = torch.empty(batchSize,3,64,64,memory_format=torch.channels_last)
static_datatape_1 = torch.empty(batchSize,3,64,64,memory_format=torch.channels_last)


def collectImgs1(sdt):
    for x in range(batchSize):
        ind = random.randint(0, filecount)
        myimg = ic.get(ind,None)
        if myimg is None:
            myimg = tv_tensors.Image(crop_center_square(Image.open(filelist[ind])).resize((64,64), Image.Resampling.LANCZOS))
            ic[ind] = myimg
        sdt.select(0,x).copy_(myimg,non_blocking = True)
        myimg = None
def collectImgs(sdt):
    collectImgs1(sdt)
    sdt.div_(255/sqrt12).sub_(0.5*sqrt12)



def mkdfs():
    with torch.no_grad():
        refrand.normal_(0.0,1.0)
        return generator_trace(refrand)



static_thing_2 = torch.empty(batchSize,1,1,1)

def ipsf(x):
    v = x[:(x.size(0) // 2)]
    v.copy_(torch.flip(v, [-1]))
    v = None
    x = None

def ipsf2(x):
    ss = (x.size(0) // 4)
    v = x[ss:-ss]
    v.copy_(torch.flip(v, [-1]))
    v = None
    x = None

def interpolate() -> torch.Tensor:
    collectImgs(static_datatape_1)
    ipsf(static_datatape_1)
    dsf = mkdfs()
    ipsf2(dsf)
    static_thing_2.uniform_()
    
    dsf.mul_(static_thing_2)
    static_thing_2.sub_(1)
    dsf.addcmul_(static_datatape,static_thing_2,value=-1)
    return dsf

#HACK: We create this special wrapper module around the discriminator
#so we can compute & backpropagate the gradient penalty with AOT Autograd
#since the gradient penalty is now part of the module itself, AOT Autograd
#will be able to optimize a lot better
class GradientPenaltyDiscriminator(torch.nn.Module):
    def __init__(self,discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, input):
        sz0_ = input.size(0)
        input.requires_grad_(True)
        graddx = torch.autograd.grad(outputs=self.discriminator.forward(input), inputs=input,create_graph=True, retain_graph=True)[0]
        input.requires_grad_(False)
        input = None
        graddx = graddx.mul(graddx)
        graddx = graddx.sum((-1,-2,-3))
        graddx = graddx.sqrt()
        graddx = graddx.sub(1.0)
        graddx = graddx.relu()
        graddx = graddx.mul(graddx)
        graddx = graddx.sum()
        return graddx
gradient_penalty_discriminator = memory_efficient_fusion(GradientPenaltyDiscriminator(discriminator_softplus))
gradient_penalty_discriminator_relu = memory_efficient_fusion(GradientPenaltyDiscriminator(discriminator_relu))

l2_regularization = 0.01


for i in range(0,360000,1):
    collectImgs(static_datatape)
    ipsf(static_datatape)
    loss3 = discriminator_trace_softplus(static_datatape).div(batchSize)
    loss3.backward()
    loss3 = loss3.tolist()
    dfks = mkdfs()
    ipsf(dfks)
    loss2 = discriminator_trace_softplus(dfks).div(-batchSize)
    loss2.backward()
    loss2 = loss2.tolist()

    print("Batch #" + str(i) + " discriminator Wasserstein loss (softplus): " + str(loss2 + loss3))



    
    itp = interpolate()
    gradientPenalty = gradient_penalty_discriminator(itp)
    gp1 = gradientPenalty.tolist()
    if gp1>0.0:
        gradientPenalty.mul(gpw/batchSize).backward()
    gradientPenalty=None
    print("Batch #" + str(i) + " discriminator gradient penalty (softplus): " + str(gp1/batchSize))
    
    for x in discriminator_softplus.parameters():
        x.requires_grad_(False)
        if x.dim() > 1:
            x.grad.add_(x,alpha=l2_regularization)
    loss3 = discriminator_trace_relu(static_datatape).div(batchSize)
    loss3.backward()
    loss3 = loss3.tolist()
    loss2 = discriminator_trace_relu(dfks).div(-batchSize)
    dfks = None
    loss2.backward()
    loss2 = loss2.tolist()

    print("Batch #" + str(i) + " discriminator Wasserstein loss (CReLU): " + str(loss2 + loss3))


    
    #OPTIMIZATION: Disable grad for biases before gradient penalty backpropagation
    #because they have no impact on gradient penalty
    for x in discriminator_relu.parameters():
        if x.dim() == 1:
            x.requires_grad_(False)
    gradientPenalty = gradient_penalty_discriminator_relu(itp)
    itp = None
    gp1 = gradientPenalty.tolist()
    if gp1>0.0:
        gradientPenalty.mul(gpw/batchSize).backward()
    gradientPenalty=None
    print("Batch #" + str(i) + " discriminator gradient penalty (CReLU): " + str(gp1/batchSize))
    
    #Adjust gradients for weight normalization
    for x in discriminator_relu.parameters():
        x.requires_grad_(False)
        if(x.dim() == 4):
            g = x.grad
            
            g.addcmul_(x, x.mul(g).sum((1,2,3),keepdim=True),value=-1)
            g = None
    optimizer_to_cuda(discriminator_optimizer)
    discriminator_optimizer.step()
    optimizer_to_cpu(discriminator_optimizer)
    discriminator_optimizer.zero_grad(set_to_none=True)
    rnrd()

    discriminator.train(False)
    generator.train(True)
    
    
    refrand.normal_(0.0,1.0)
    loss1 = generator_discriminator(refrand)
    loss1.backward()
    loss1 = loss1.tolist()
    

    
    print("Batch #" + str(i) + " generator Wasserstein loss: " + str(loss1))
    
    
    if i % 100 == 0:
        #with torch.no_grad():
            #save_image(generator.forward(torch.randn(1,2048))[0].contiguous().div_(sqrt12).add_(0.5).squeeze(0), "fakecats/cat" + str(i) + ".png")
        dumpgrad(generator)


    optimizer_to_cuda(generator_optimizer)
    generator_optimizer.step()
    optimizer_to_cpu(generator_optimizer)
    generator_optimizer.zero_grad(set_to_none=True)



    for x in discriminator.parameters():
        x.requires_grad_(True)
    
    generator.train(False)
    discriminator.train(True)
    
    
    
    print()
    


    
    

torch.save(generator.state_dict(),"models/generator")
torch.save(generator_optimizer.state_dict(),"models/generator_optimizer")
torch.save(discriminator.state_dict(),"models/discriminator")
torch.save(discriminator_optimizer.state_dict(),"models/discriminator_optimizer")


for x in discriminator.parameters():
    x.requires_grad_(False)

for x in generator.parameters():
    x.requires_grad_(False)


imgss = generator.forward(torch.randn(256,2048))

imgss.requires_grad_(True)
discriminator_relu.forward(brfmod.forward(imgss)).backward()
discriminator_softplus.forward(brfmod.forward(imgss)).backward()
imgss.requires_grad_(False)

imgss.grad.div_(imgss.grad.mul(imgss.grad).mean((-1,-2,-3),keepdim=True).sqrt_())

#print(imgss.grad.mul(imgss.grad).sum((-1,-2,-3)).sqrt_().div_(((64*64*3)//256)).tolist())


imgss = torch.cat([imgss,imgss.grad],-1)
imgss.div_(sqrt12).add_(0.5)
imgss = torch.nn.functional.pad(imgss,(2,2,2,2))

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

save_image(imgss, "fakecats1.png")
