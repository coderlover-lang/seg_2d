import torch 
import torch.nn as nn 
from torch.nn import functional as  F 



class MiddleModule(nn.Module):
    def __init__(self, proj_dim = 64):
        super().__init__()
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(512 * 2, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(256 * 2, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(128 * 2, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(64 * 2, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        
        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim),
                                         nn.ReLU(inplace=True))
        
        self.level3 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(512 * 2, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(proj_dim, 1, kernel_size=3, padding=1))
        
    def forward(self, X):
        '''
        X : [bs 64, 32, 32], [bs, 128, 16, 16], [bs, 256, 8, 8], [bs, 512, 4, 4]
        '''
        x2, x3, x4, x5 = X 
        
        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1,size=x4.size()[2:], mode='bilinear')-x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1,size=x3.size()[2:], mode='bilinear')-x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1,size=x2.size()[2:], mode='bilinear')-x2_dem_1))
        # x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1,size=x1.size()[2:], mode='bilinear')-x1))


        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        # x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))


        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        # x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        # x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        # level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        # output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)

        # output = F.upsample(output2, size=(128, 128), mode='bilinear')
        
        return (output2, output3, output4)
if __name__ == "__main__":
    mm = MiddleModule()
    bs = 8 
    X = [torch.rand(*i) for i in [[bs, 64, 32, 32], [bs, 128, 16, 16], [bs, 256, 8, 8], [bs, 512, 4, 4]]]
    print([i.shape for i in X])
    print([i.shape for i in mm(X)])