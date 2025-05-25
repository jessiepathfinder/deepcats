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
#from torchvision.transforms import Resize,InterpolationMode


cuda = torch.device('cuda')
torch.set_default_device(cuda)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)


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




class VariancePreservingForceDropout2D(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.dropout = 1.0 - dropout
        self.donorm = math.sqrt(1.0 - dropout)

    def forward(self, input):
        return input.mul(torch.rand(list(input.size()[:-2]) + [1,1],dtype=input.dtype,device=input.device).bernoulli_(self.dropout).div_(self.donorm))


        
class SelfAttentionThingV2(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        gain = math.sqrt(size)
        self.keys = torch.nn.Parameter(torch.randn(size,size).div_(gain))
        self.queries = torch.nn.Parameter(torch.randn(size,size).div_(gain))

    def forward(self, input):
        x = input.transpose(-1,-3)
        x = x.flatten(-3,-2)
        
        x1 = x.unsqueeze(-3)

        x = x.div(x.mul(x).mean((-2,-1),keepdim=True).sqrt())
        
        x = x.unsqueeze(-3)
        (x,y) = (x.matmul(self.queries), x.matmul(self.keys))
        x = torch.nn.functional.scaled_dot_product_attention(x,y,x1)
        x1 = None
        y = None
        x = x.squeeze(-3)
        x = x.unflatten(-2,input.size()[-2:])
        x = x.transpose(-1,-3)
        return torch.cat([input,x],-3).to(memory_format=torch.channels_last)



class FastAdaBelief:
    def __init__(self,parameters,lr=1e-8,beta1=0.9,beta2=0.999,epsilon = 1e-6,sadam_fallback = False):
        self.state = [(x,torch.zeros_like(x),torch.zeros_like(x)) for x in parameters]
        self.stepnr = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.sadam_fallback = sadam_fallback
    def step(self):
        stepnr = self.stepnr + 1
        self.stepnr = stepnr
        b1 = self.beta1
        b1_ = 1.0 - b1
        b2 = self.beta2
        corr = (1.0 - math.pow(b2,stepnr))
        epsilon = self.epsilon * (1.0 - b2) * corr
        nlr = (self.lr * corr) / (b2 - 1.0)
        sadam_fallback = self.sadam_fallback
        with torch.no_grad():
            for (parameter,exp_avg,exp_avg_sq) in self.state:
                grad = parameter.grad
                parameter.grad = None
                exp_avg.mul_(b1)
                exp_avg.add_(grad,alpha=b1_)
                exp_avg_sq.mul_(b2)
                grad = grad.mul(grad) if sadam_fallback else torch.nn.functional.mse_loss(grad,exp_avg,reduction="none")
                exp_avg_sq.add_(grad)
                grad = None
                parameter.addcdiv_(exp_avg,exp_avg_sq.add(epsilon),value=nlr)

class BiasLayer(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input):
        return input.add(self.bias)
            
class HalfNormInstance2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sub(input.mean((-1,-2),keepdim=True))

class WeightNormLayer2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.div(input.mul(input).mean((-1,-2,-3),keepdim=True).sqrt())

class NSplit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cat([input,input.neg()],-3)

class PolynomialKernelTrick(torch.nn.Module):
    def __init__(self,howmuch : int):
        super().__init__()
        self.howmuch = howmuch
    def forward(self,input):
        queue = [input]
        y = input
        for x in range(self.howmuch):
            y = y.mul(input)
            queue.append(y)
        return torch.cat(queue,-3)
            


arluv2_mod = ArLUv2(0.0)
arluv2_flat_mod = ArLUv2_flat(0.0)
arluv2_flat_mod_dropout = torch.jit.script(ArLUv2_flat(0.25))
arluv2_mod_dropout = torch.jit.script(ArLUv2(0.25))

sqrt12 = math.sqrt(12)


        


discriminator = torch.nn.Sequential(
PolynomialKernelTrick(3),
convinit3nb(biasinit(torch.nn.Conv2d(12,128,4,stride=2,padding=1,padding_mode="replicate")),16),arluv2_mod_dropout,
convinit3nb(biasinit(torch.nn.Conv2d(256,256,4,stride=2,padding=1)),16),arluv2_mod_dropout,
convinit3nb(biasinit(torch.nn.Conv2d(512,512,4,padding=1,stride=2)),16),arluv2_mod_dropout,
convinit3nb(biasinit(torch.nn.Conv2d(1024,1024,4,padding=1,stride=2)),16),arluv2_mod_dropout,
Transpose(-3,-1),Transpose(-2,-1),torch.nn.Flatten(-3,-1),
makekaiminglinear(32768,1,False)
)




fdinitgain = 1.0 / 5.0





#generator_initstd = 0.02540494454



attn_query_gain = 1.0

fdo_mod = torch.jit.script(VariancePreservingForceDropout2D(0.25))
fastblur_mod = torch.nn.AvgPool2d(2,stride=1)


generator = torch.nn.Sequential(
    makekaiminglinear(4096,4096),arctan_mod,
    makekaiminglinear(4096,4096),arctan_mod,
    makekaiminglinear(4096,16384),arctan_mod,
    torch.nn.Unflatten(-1, (4,4,1024)),Transpose(-3,-1),Transpose(-2,-1),fdo_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(1024,512,4,padding=1,stride=2)),4),arctan_mod,fdo_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(512,256,4,padding=1,stride=2)),4),arctan_mod,fdo_mod,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(256,128,4,padding=1,stride=2)),4),arctan_mod,fdo_mod,
    convinit3nb(biasinit(torch.nn.Conv2d(128,128,3,padding=1)),9),arctan_mod,fdo_mod,SelfAttentionThingV2(128),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9),arctan_mod,fdo_mod,SelfAttentionThingV2(128),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9),arctan_mod,fdo_mod,SelfAttentionThingV2(128),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9),arctan_mod,fdo_mod,SelfAttentionThingV2(128),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9),arctan_mod,fdo_mod,SelfAttentionThingV2(128),
    convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9),arctan_mod,WeightNormLayer2D(),
    convinit3nb(torch.nn.Conv2d(128,12,4,padding=2,bias=False),16),torch.nn.PixelShuffle(2),fastblur_mod,fastblur_mod,BiasLayer((3,1,1))
)




discriminator.train(True)
generator.train(False)
generator_trace = torch.jit.trace(generator,torch.empty(batchSize,4096),check_trace=False)


discriminator_trace = memory_efficient_fusion(torch.nn.Sequential(FlipAugment(),discriminator,mean_mod))
generator.train(True)
discriminator.train(False)


generator_discriminator = memory_efficient_fusion(torch.nn.Sequential(generator,FlipAugment1(), discriminator,mean_mod))
discriminator.train(True)
generator.train(False)







filelist = []


for currentpath, folders, files in os.walk("."):
    for file in files:
        if file.endswith(".jpg"):
            filelist.append(os.path.join(currentpath, file))

filecount = len(filelist) - 1
        



generator_optimizer = torch.optim.Adam(generator.parameters(),lr=1e-5,eps=1e-9)
#generator_optimizer = FastAdaBelief(generator.parameters(),lr=1e-7,epsilon=8.13802083e-5,sadam_fallback=True)
discriminator_optimizer = adabelief_pytorch.AdaBelief(discriminator.parameters(),lr=1e-4,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)
#discriminator_optimizer = FastAdaBelief(discriminator.parameters(),lr=1e-7,epsilon=8.13802083e-5)



gpw = math.sqrt(64*64*3)

def bRandFlip(input : torch.Tensor):
    size = input.size(0)
    flipIndices = []
    noFlipIndices = []
    indList = (flipIndices, noFlipIndices)
    for x in range(size):
        indList[random.randint(0,1)].append(x)
    l = len(flipIndices)
    if l == 0:
        return input
    if l == size:
        return torch.flip(input,[-1])
    return torch.cat([input[noFlipIndices], torch.flip(input[flipIndices], [-1])],0)


for x in generator.parameters():
    x.requires_grad_(True)



def dumpgrad(mod):
    with torch.no_grad():
        gsd = mod.state_dict(keep_vars=True)
        for x in mod.state_dict(keep_vars=True):
            mygrad = gsd[x].grad
            print(x + ": " + str(mygrad.mul(mygrad).mean().sqrt().tolist()))

ic = {}

static_datatape = torch.empty(batchSize,3,64,64,memory_format=torch.channels_last)


def collectImgs():
    for x in range(batchSize):
        ind = random.randint(0, filecount)
        myimg = ic.get(ind,None)
        if myimg is None:
            myimg = tv_tensors.Image(Image.open(filelist[ind]).resize((64,64), Image.Resampling.LANCZOS))
            ic[ind] = myimg
        static_datatape.select(0,x).copy_(myimg,non_blocking = True)
        myimg = None
    static_datatape.div_(255/sqrt12).sub_(0.5*sqrt12)

l2_regularization = 1e-3

refrand = torch.empty(batchSize,4096)
def mkdfs():
    with torch.no_grad():
        refrand.normal_(0.0,1.0)
        return generator_trace(refrand)



def interpolate() -> torch.Tensor:
    vec = torch.rand(batchSize,1,1,1)
    return bRandFlip(bRandFlip(mkdfs()).mul_(vec).addcmul_(static_datatape,vec.sub_(1),value=-1))

#HACK: We create this special wrapper module around the discriminator
#so we can compute & backpropagate the gradient penalty with AOT Autograd
#since the gradient penalty is now part of the module itself, AOT Autograd
#will be able to optimize a lot better

#this also bypasses the AOT Autograd limitation of not being able to compute second derivatives
class GradientPenaltyDiscriminator(torch.nn.Module):
    def __init__(self,discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, input):
        input.requires_grad_(True)
        graddx = torch.autograd.grad(outputs=self.discriminator.forward(input).sum(), inputs=input,create_graph=True, retain_graph=True)[0]
        input.requires_grad_(False)
        input = None
        graddx = graddx.mul(graddx)
        graddx = graddx.sum((1,2,3))
        graddx = graddx.sqrt()
        graddx = graddx.sub(1.0)
        graddx = graddx.mul(graddx)
        graddx = graddx.mean()
        return graddx
gradient_penalty_discriminator = aot_module(GradientPenaltyDiscriminator(discriminator),fw_compiler=ts_compile,bw_compiler=ts_compile, partition_fn=min_cut_rematerialization_partition)






# maxlr = 1e-4
# alr = 1e-6
# target_log_delta_square = math.log(1e-4)
# lr_update_rate = 0.01

for i in range(100001):
    collectImgs()
    loss3 = discriminator_trace(static_datatape)
    loss3.backward()
    loss2 = discriminator_trace(mkdfs()).neg()
    loss2.backward()
    print("Batch #" + str(i) + " discriminator Wasserstein loss: " + str(loss2.tolist() + loss3.tolist()))
    collectImgs()
    gradientPenalty = gradient_penalty_discriminator(interpolate())
    gradientPenalty.mul(gpw).backward()
    
    print("Batch #" + str(i) + " discriminator gradient penalty: " + str(gradientPenalty.tolist()))
    
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)
    
    
    
    
    

    
    for x in discriminator.parameters():
        x.requires_grad_(False)


    
    discriminator.train(False)
    generator.train(True)
    refrand.normal_(0.0,1.0)
    loss1 = generator_discriminator(refrand)
    loss1.backward()
    
    print("Batch #" + str(i) + " generator Wasserstein loss: " + str(loss1.tolist()))
    dumpgrad(generator)
    
    generator_optimizer.step()
    generator_optimizer.zero_grad(set_to_none=True)
    
    
    

    print()
    generator.train(False)
    discriminator.train(True)
    for x in discriminator.parameters():
        x.requires_grad_(True)

    
    
    if i % 100 == 0:
        with torch.no_grad():
            save_image(generator.forward(torch.randn(1,4096)).div_(sqrt12).add_(0.5).squeeze(0), "fakecats/cat" + str(i) + ".png")
        if i % 10000 == 0:
            torch.save(generator.state_dict(),"models/generator_" + str(i))
            #torch.save(generator_optimizer.state_dict(),"models/generator_optimizer_" + str(i))
            #torch.save(discriminator.state_dict(),"models/discriminator_" + str(i))
            #torch.save(discriminator_optimizer.state_dict(),"models/discriminator_optimizer_" + str(i))

    
    

