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
from functorch.compile import nop as no_compiler
#from torchvision.transforms import Resize,InterpolationMode


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

class Arctan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.atan()


class Softplusv2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.softplus(torch.cat([input,input.neg()],-3))

class Softplusv2_flat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.softplus(torch.cat([input,input.neg()],-1))

class ForceDropout2D(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        dropout = 1.0 - dropout
        self.dropout = dropout

    def forward(self, input):
        return input.mul(torch.rand(list(input.size()[:-2]) + [1,1],dtype=input.dtype,device=input.device).bernoulli_(self.dropout))

class ForceDropout(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        dropout = 1.0 - dropout
        self.dropout = dropout
        self.makeup_gain = math.sqrt(dropout)

    def forward(self, input):
        return input.mul(torch.rand_like(input).bernoulli_(self.dropout))

softplus_mod = torch.nn.Softplus()
fdo_mod = torch.jit.script(ForceDropout2D(0.125))
fdo_mod_1 = torch.jit.script(ForceDropout(0.125))




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







class BiasLayer(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input):
        return input.add(self.bias)
            

class ForkAdd(torch.nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, input):
        return self.a.forward(input).add(self.b.forward(input))

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

class NormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.div(input.mul(input).mean((-1),keepdim=True).sqrt())
class NormLayer2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, input):
        return input.div(input.mul(input).mean((-1,-2,-3),keepdim=True).sqrt())
        
class ResBlock(torch.nn.Module):
    def __init__(self,siz):
        super().__init__()
        self.i = biasinit(convinit3nb(torch.nn.Conv2d(siz,siz,3,padding=2),9))
        self.o = biasinit(convinit3nb(torch.nn.Conv2d(siz,siz,3),9))
        self.do = fdo_mod
        self.no = norm_layer
    def forward(self,input):
        return input.add(self.o.forward(self.no.forward(self.do.forward(self.i.forward(input).atan()))))



norm_layer = NormLayer2d()



norm_layer_1 = NormLayer()


sqrt12 = math.sqrt(12)


        
softplusv2_mod = Softplusv2()
softplusv2_flat_mod = Softplusv2_flat()

fastblur_mod = torch.nn.AvgPool2d(2,stride=1)



discriminator = ForkAdd(
    torch.nn.Sequential(
        PolynomialKernelTrick(5),
        convinit3nb(biasinit(torch.nn.Conv2d(18,128,3)),9),arctan_mod,

        convinit3nb(biasinit(torch.nn.Conv2d(128,128,4,stride=2,padding=2)),16),softplusv2_mod,
        convinit3nb(biasinit(torch.nn.Conv2d(256,256,4,stride=2,padding=1)),16),softplusv2_mod,
        convinit3nb(biasinit(torch.nn.Conv2d(512,512,4,padding=1,stride=2)),16),softplusv2_mod,
        convinit3nb(biasinit(torch.nn.Conv2d(1024,1024,4,padding=1,stride=2)),16),softplusv2_mod,
        Transpose(-3,-1),Transpose(-2,-1),torch.nn.Flatten(-3,-1),
        makekaiminglinear2(32768,1,False),ConstantDiv(math.sqrt(32768*2))        
    ),
    torch.nn.Sequential(
        torch.nn.AvgPool2d(2),
        Transpose(-3,-1),Transpose(-2,-1),torch.nn.Flatten(-3,-1),
        makekaiminglinear(3072,3072),softplusv2_flat_mod,
        makekaiminglinear(6144,3072),softplusv2_flat_mod,
        makekaiminglinear(6144,3072),softplusv2_flat_mod,
        makekaiminglinear(6144,3072),softplusv2_flat_mod,
        makekaiminglinear(6144,3072),softplusv2_flat_mod,
        makekaiminglinear2(6144,1,False),ConstantDiv(math.sqrt(6144*2))
    )
)





fdinitgain = 1.0 / 5.0





#generator_initstd = 0.02540494454



attn_query_gain = 1.0





generator = torch.nn.Sequential(
    makekaiminglinear(4096,4096),arctan_mod,fdo_mod_1,norm_layer_1,
    makekaiminglinear(4096,4096),arctan_mod,fdo_mod_1,norm_layer_1,
    makekaiminglinear(4096,16384),arctan_mod,
    torch.nn.Unflatten(-1, (4,4,1024)),Transpose(-3,-1),Transpose(-2,-1),fdo_mod,norm_layer,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(1024,512,4,padding=1,stride=2)),4),arctan_mod,norm_layer,
    ResBlock(512),norm_layer,
    ResBlock(512),norm_layer,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(512,256,4,padding=1,stride=2)),4),arctan_mod,norm_layer,
    ResBlock(256),norm_layer,
    ResBlock(256),norm_layer,
    convinit2nb(biasinit(torch.nn.ConvTranspose2d(256,128,4,padding=1,stride=2)),4),arctan_mod,norm_layer,
    ResBlock(128),norm_layer,
    ResBlock(128),norm_layer,
    convinit3nb(torch.nn.Conv2d(128,12,3,padding=1,bias=False),9),torch.nn.PixelShuffle(2),BiasLayer((3,1,1))
)

l2_regularization = 0.01


discriminator.train(True)
generator.train(False)
generator_trace = torch.jit.trace(generator,torch.empty(batchSize,4096),check_trace=False)


fa_mod = FlipAugment()

discriminator_trace = memory_efficient_fusion(torch.nn.Sequential(fa_mod,discriminator,mean_mod))

generator.train(True)
discriminator.train(False)

generator_discriminator = memory_efficient_fusion(torch.nn.Sequential(generator,fa_mod, discriminator,mean_mod))
discriminator.train(True)
generator.train(False)







filelist = []


for currentpath, folders, files in os.walk("."):
    for file in files:
        if file.endswith(".jpg"):
            filelist.append(os.path.join(currentpath, file))

filecount = len(filelist) - 1
        



generator_optimizer = torch.optim.Adam(generator.parameters(),lr=1e-5,eps=1e-9)
#generator_optimizer = torch.optim.SGD(generator.parameters(),lr=1e-5,momentum=0.9)
discriminator_optimizer = adabelief_pytorch.AdaBelief(discriminator.parameters(),lr=1e-4,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)



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

def collectImgs1():
    for x in range(batchSize):
        ind = random.randint(0, filecount)
        myimg = ic.get(ind,None)
        if myimg is None:
            myimg = tv_tensors.Image(crop_center_square(Image.open(filelist[ind])).resize((64,64), Image.Resampling.LANCZOS))
            ic[ind] = myimg
        static_datatape.select(0,x).copy_(myimg,non_blocking = True)
        myimg = None
def collectImgs():
    collectImgs1()
    static_datatape.div_(255/sqrt12).sub_(0.5*sqrt12)



refrand = torch.empty(batchSize,4096)
def mkdfs():
    with torch.no_grad():
        refrand.normal_(0.0,1.0)
        return generator_trace(refrand)

gpw1 = math.sqrt(16*16*3)

def interpolate1() -> torch.Tensor:
    vec = torch.rand(batchSize,1,1,1)
    return bRandFlip(bRandFlip(torch.nn.functional.avg_pool2d(mkdfs(),2,divisor_override=2)).mul_(vec).addcmul_(torch.nn.functional.avg_pool2d(static_datatape,2,divisor_override=2).div_(255/sqrt12).sub_(0.5*sqrt12),vec.sub_(1),value=-1))

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
        graddx = graddx.sum((-1,-2,-3))
        graddx = graddx.sqrt()
        graddx = graddx.sub(1.0)
        graddx = graddx.mul(graddx)
        graddx = graddx.mean()
        return graddx


gradient_penalty_discriminator = aot_module(GradientPenaltyDiscriminator(discriminator),fw_compiler=ts_compile,bw_compiler=ts_compile, partition_fn=min_cut_rematerialization_partition)



gpw2 = gpw * 2.0


# maxlr = 1e-4
# alr = 1e-6
# target_log_delta_square = math.log(1e-4)
# lr_update_rate = 0.01

for i in range(200001):
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
    with torch.no_grad():
        for x in discriminator.parameters():
            if(x.dim() < 2):
                continue
            l2z = l2_regularization
            if x.size(0) == 1:
                l2z /= math.sqrt(x.size(1))
            x.grad.add_(x,alpha=l2z)
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


    
    

