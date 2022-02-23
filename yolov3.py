import torch
import torchvision
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CBL(nn.Module): 
    def __init__(self, c1, c2, k=1, s=2, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,padding=p,groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self,x):
        return self.activate(self.bn(self.conv(x)))

class ResUint(nn.Module):
    def __init__(self, c1, c2,shortcut = True, e = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cbl1 = CBL(c1=c1,c2=c_,k=1,s=1,p=autopad(1))
        self.cbl2 = CBL(c1=c_,c2=c2,k=3,s=1,p=autopad(3))
        self.add = shortcut and c1 == c2

    def forward(self,x):
        return x + self.cbl2(self.cbl1(x)) if self.add else self.cbl2(self.cbl1(x))


class ResX(nn.Module):
    def __init__(self,c1,c2,n=1,chortcut = True,e=0.5):
        super().__init__()
        self.cbl = CBL(c1,c2,s=2)
        self.m = nn.Sequential(*( ResUint(c2,c2,chortcut,0.5) for i in range(n)))

    def forward(self,x):
        return self.m(self.cbl(x))



class YOLOV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.CBL = CBL(3,32,s=1)
        self.Res1 = ResX(32,64,n=1)
        self.Res2 = ResX(64,128,n=2)
        self.Res8_1 = ResX(128,256,n=8)
        self.Res8_2 = ResX(256,512,n=8)
        self.Res4 = ResX(512,1024, n=4)


        self.Out1Cbl1_5 = nn.Sequential(
            CBL(1024,512,k=1,s=1,p=autopad(1)),
            CBL(512,1024,k=3,s=1,p=autopad(3)),
            CBL(1024,512,k=1,s=1,p=autopad(1)),
            CBL(512,1024,k=3,s=1,p=autopad(3)),
            CBL(1024,512,k=1,s=1,p=autopad(1)),
            )
        
        self.Out1Cbl2 = CBL(512,1024,k=3,s=1,p=autopad(3))

        self.OutConv1 = nn.Conv2d(1024,255,kernel_size=1,stride=1,padding=autopad(1))


        self.UpCbl1 = CBL(512,256,k=1,s=1,p=autopad(1))

        self.UpSample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.Out2Cbl1_5 = nn.Sequential(
            CBL(768,256,k=1,s=1,p=autopad(1)),
            CBL(256,512,k=3,s=1,p=autopad(3)),
            CBL(512,256,k=1,s=1,p=autopad(1)),
            CBL(256,512,k=3,s=1,p=autopad(3)),
            CBL(512,256,k=1,s=1,p=autopad(1)),
            )
        self.Out2Cbl2 = CBL(256,512,k=3,s=1,p=autopad(3))

        self.OutConv2 = nn.Conv2d(512,255,kernel_size=1,stride=1,padding=autopad(1))


        self.UpCbl2 = CBL(256,128,k=1,s=1,p=autopad(1))

        self.UpSample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.Out3Cbl1_5 = nn.Sequential(
            CBL(384,128,k=1,s=1,p=autopad(1)),
            CBL(128,256,k=3,s=1,p=autopad(3)),
            CBL(256,128,k=1,s=1,p=autopad(1)),
            CBL(128,256,k=3,s=1,p=autopad(3)),
            CBL(256,128,k=1,s=1,p=autopad(1)),
            )
        self.Out3Cbl2 = CBL(128,256,k=3,s=1,p=autopad(3))

        self.OutConv3 = nn.Conv2d(256,255,kernel_size=1,stride=1,padding=autopad(1))
    def forward(self,x):

        # 32
        x = self.CBL(x)

        # 256
        x1 = self.Res8_1(self.Res2(self.Res1(x)))

        # 512
        x2 = self.Res8_2(x1)

        # 1024
        x3 = self.Res4(x2)

        out1_ = self.Out1Cbl1_5(x3)
        out1  = self.OutConv1(self.Out1Cbl2(out1_))
        out1_ = self.UpSample1(self.UpCbl1(out1_))

        out2_ = torch.cat((out1_,x2),dim=1)
        out2_ = self.Out2Cbl1_5(out2_)
        out2 = self.OutConv2(self.Out2Cbl2(out2_))
        out2_ = self.UpSample2(self.UpCbl2(out2_))

        out3_ = torch.cat((out2_,x1),dim=1)
        out3  = self.OutConv3(self.Out3Cbl2(self.Out3Cbl1_5(out3_)))



        return out1,out2,out3


if __name__ == "__main__":


    net = YOLOV3()

    a = torch.ones(size=(1,3,416,416))

    out1,out2,out3 = net(a)

    print(out1.shape,out2.shape,out3.shape)


