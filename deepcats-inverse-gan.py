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
        
class VariancePreservingDropout(torch.nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.dropout = 1.0 - dropout
        self.donorm = math.sqrt(1.0 - dropout)

    def forward(self, input):
        if self.training:
            return input
        return input.mul(torch.rand_like(input).bernoulli_(self.dropout).div_(self.donorm))




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





class WeightNormalizingLinear(torch.nn.Module):
    def __init__(self, weight : torch.Tensor):
        super().__init__()
        self.weight = weight
    def forward(self,input : torch.Tensor):
        w = self.weight
        w = w.div(w.mul(w).sum().sqrt())
        w = w.transpose(0,1)
        return input.matmul(w)
        
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

class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.div(input.mul(input).mean(-1,keepdim=True).sqrt())

class BiasLayer(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input):
        return input.add(self.bias)

class BatchHalfNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sub(input.mean(0,keepdim=True))

arluv2_mod = ArLUv2(0.0)
arluv2_flat_mod = ArLUv2_flat(0.0)
arluv2_flat_mod_dropout = torch.jit.script(ArLUv2_flat(0.25))


generator_gain = 1.48663410298

attn_query_gain = 1.0 / 0.90747


encoder = torch.nn.Sequential(
ConstantMul(0.90747),
AugmentKernel(),
convinit3nb(biasinit(torch.nn.Conv2d(12,128,4,stride=2)),16,gain=generator_gain),arctan_mod,
convinit3nb(biasinit(torch.nn.Conv2d(128,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,SelfAttentionThing(128,attn_query_gain),
convinit3nb(biasinit(torch.nn.Conv2d(256,128,3,padding=1)),9,gain=generator_gain),arctan_mod,
convinit3nb(biasinit(torch.nn.Conv2d(128,256,4,padding=1,stride=2)),16,gain=generator_gain),arctan_mod,
convinit3nb(biasinit(torch.nn.Conv2d(256,512,4,padding=1,stride=2)),16,gain=generator_gain),arctan_mod,
convinit3nb(biasinit(torch.nn.Conv2d(512,1024,4,padding=1,stride=2)),16,gain=generator_gain),arctan_mod,
Transpose(-3,-1),Transpose(-2,-1),torch.nn.Flatten(-3,-1),
makekaiminglinear(16384,4096,gain=generator_gain),arctan_mod,
makekaiminglinear(4096,4096,gain=generator_gain),arctan_mod,Normalize(),
makekaiminglinear(4096,4096)
)




decoder = torch.nn.Sequential(
    ConstantMul(0.90747),
    makekaiminglinear(4096,4096,gain=generator_gain),arctan_mod,
    makekaiminglinear(4096,4096,gain=generator_gain),arctan_mod,
    makekaiminglinear(4096,16384,gain=generator_gain),arctan_mod,
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




l2_regularization = 1e-3
l2_regularization_final = l2_regularization / math.sqrt(8192)


discriminator_upper = [
makekaiminglinear(4096,4096,True),arluv2_flat_mod_dropout,
makekaiminglinear(8192,4096,True),arluv2_flat_mod_dropout,
makekaiminglinear(8192,4096,True),arluv2_flat_mod_dropout,
makekaiminglinear(8192,4096,True),arluv2_flat_mod_dropout,
makekaiminglinear(8192,4096,True),arluv2_flat_mod_dropout

]

discriminator_final = makekaiminglinear2(8192,1,True)

discriminator = torch.nn.Sequential(
*discriminator_upper,discriminator_final,ConstantDiv(math.sqrt(8192))
)
discriminator_upper = torch.nn.ModuleList(discriminator_upper)


encoder_trace = memory_efficient_fusion(encoder)
decoder_trace = memory_efficient_fusion(decoder)

discriminator.train(False)
discriminator_trace_eval = memory_efficient_fusion(torch.nn.Sequential(discriminator,torch.nn.Softplus(),mean_mod))

discriminator.train(True)
discriminator_trace = memory_efficient_fusion(discriminator)





cnoise = torch.randn(1,4096)

def dumpgrad(mod):
    with torch.no_grad():
        gsd = mod.state_dict(keep_vars=True)
        for x in mod.state_dict(keep_vars=True):
            mygrad = gsd[x].grad
            print(x + ": " + str(mygrad.mul(mygrad).mean().sqrt().tolist()))

#encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=0.01,momentum=0.9)
encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=1e-5,eps=1e-9)
#encoder_optimizer = adabelief_pytorch.AdaBelief(encoder.parameters(),lr=1e-4,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)

decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=0.01,momentum=0.9)
discriminator_optimizer = adabelief_pytorch.AdaBelief(discriminator.parameters(),lr=1e-4,degenerated_to_sgd=False,eps=1e-9,weight_decouple=False,rectify=False,print_change_log=False)
#discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=1e-4,eps=1e-9)


#encoder_upper_optimizer = torch.optim.SGD(encoder_upper.parameters(),lr=1e-1,momentum=0.9)

imgsize = 64*64*3

#encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=1e-1/imgdivide,momentum=0.9)
#decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=1e-1,momentum=0.9)

def makehelinear(lin):
    with torch.no_grad():
        lin.weight.abs_()
    return lin


def L2Regularize(model, gamma):
    with torch.no_grad():
        for x in model.parameters():
            i = 0
            for y in x.size():
                if y > 0:
                    ++i
            if(i > 1):
                x.grad.add_(x, gamma)



ic = {}

static_datatape = torch.empty(batchSize,3,64,64,memory_format=torch.channels_last)

siglv = static_datatape[:(batchSize//2)]

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
    siglv.copy_(torch.flip(siglv,[-1]),non_blocking=True)

torch.autograd.grad_mode.inference_mode(True)
encoder_trace_eval = torch.jit.trace(encoder,static_datatape,check_trace=False)
torch.autograd.grad_mode.inference_mode(False)

filelist = []


for currentpath, folders, files in os.walk("."):
    for file in files:
        if file.endswith(".jpg"):
            filelist.append(os.path.join(currentpath, file))
filecount = len(filelist) - 1

sqrt12 = math.sqrt(12)




discriminatorScale = math.sqrt(4096 * batchSize)

for i in range(100001):
    collectImgs()
    with torch.no_grad():
        loss4 = encoder_trace_eval(static_datatape)
    loss4 = discriminator_trace(loss4)
    loss4 = loss4.neg()
    loss4 = torch.nn.functional.softplus(loss4)
    loss4 = loss4.mean()
    
    loss4.backward()
    loss3 = discriminator_trace(torch.randn(batchSize,4096))
    loss3 = torch.nn.functional.softplus(loss3)
    loss3 = loss3.mean()
    loss3.backward()
    
    L2Regularize(discriminator_upper,l2_regularization)
    L2Regularize(discriminator_final,l2_regularization_final)
    
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)
    
    
    print("Batch #" + str(i) + " discriminator cross-entropy loss: " + str((loss4.tolist() + loss3.tolist())/2))
    
    discriminator.train(False)
        
    for x in discriminator.parameters():
        x.requires_grad_(False)
    
    collectImgs()
    
        
    encoded = encoder_trace.forward(static_datatape)
   
    #gradient normalization HACK
    encoded1 = encoded.detach()
    encoded1.requires_grad_(True)
    
    loss2 = discriminator_trace_eval(encoded1)
    loss2.backward()
    
    encoded1 = encoded1.grad
    dgradscale = 1.0 / (discriminatorScale * math.sqrt(encoded1.mul(encoded1).sum().tolist()))
    
    
    encoded1 = encoded.mul(encoded1)

    encoded1 = encoded1.sum()
    
    deepfakes = decoder_trace(encoded)
    
    
    loss1 = torch.nn.functional.l1_loss(deepfakes,static_datatape)
    if i % 100 == 0:
        with torch.no_grad():
            save_image(deepfakes.detach()[0].div(sqrt12).add_(0.5), "fakecats/reconstructed_cat" + str(i) + ".png")
    deepfakes = None
    
    
    encoded1 = loss1.add(encoded1,alpha=dgradscale)
    encoded1.backward()
    #dumpgrad(decoder)
    
    
    encoder_optimizer.step()
    encoder_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.step()
    decoder_optimizer.zero_grad(set_to_none=True)
    

    
    print("Batch #" + str(i) + " decoding loss: " + str(loss1.tolist()))
    print("Batch #" + str(i) + " inverse GAN loss: " + str(loss2.tolist()))
    
    
    if i % 100 == 0:
        with torch.no_grad():
            save_image(decoder.forward(cnoise).div_(sqrt12).add_(0.5).squeeze(0), "fakecats/cat" + str(i) + ".png")
    
    for x in discriminator.parameters():
        x.requires_grad_(True)
    
    discriminator.train(True)
    

    print()
    if i % 10000 == 0:
        torch.save(decoder.state_dict(),"models/decoder_" + str(i))
        torch.save(decoder_optimizer.state_dict(),"models/decoder_optimizer_" + str(i))
        torch.save(encoder.state_dict(),"models/encoder_" + str(i))
        torch.save(encoder_optimizer.state_dict(),"models/encoder_optimizer_" + str(i))
        torch.save(discriminator.state_dict(),"models/discriminator_" + str(i))
        torch.save(discriminator_optimizer.state_dict(),"models/discriminator_optimizer_" + str(i))
    
    
    
    

