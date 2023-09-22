import torch
import torch.nn as nn
#model
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        """spatial"""
        # 3 x 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=198, out_channels=96, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=48, out_channels=4, kernel_size=3,
                      stride=1, padding=1),
        )

        # 5 x 5
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=198, out_channels=96, kernel_size=5,
                      stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=5,
                      stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=48, out_channels=4, kernel_size=5,
                      stride=1, padding=2)
        )

        # 7 x 7
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=198, out_channels=96, kernel_size=7,
                      stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=7,
                      stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=48, out_channels=4, kernel_size=7,
                      stride=1, padding=3)
        )



        # dimensionality reduction
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=206, out_channels=96, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3,
                      stride=1, padding=1),

        )

        # -->endmember numbers
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(24),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=12, out_channels=4, kernel_size=3,
                      stride=1, padding=1),

        )

        # -->endmember numbers
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=4, kernel_size=1,
                      stride=1, padding=0)
        )



        """spectral"""
        # 3-DCNN + 2-DCNN
        self.spectral = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),
            nn.MaxPool3d((2, 1, 1)),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),
            nn.MaxPool3d((2, 1, 1)),

            nn.Conv2d(in_channels=10, out_channels=4, kernel_size=1, stride=1, padding=0)
        )
        #attention
        self.spectral_attention = nn.Sequential(
            nn.Conv2d(48, 48 // 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(48 // 2, 48, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(48, 48 // 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48 // 2, 48, kernel_size=1, stride=1)
        )




        # encoder
        self.encodelayer = nn.Sequential(nn.Softmax())

        # --> endmember numbers
        self.transconv3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1, stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
        )

        #decoder
        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )
        self.decoderlayer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )
        self.decoderlayer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

    def forward(self, x):
        #spatial branch
        layer3out = self.layer3(x)  # 7x7
        layer2out = self.layer2(x)  # 5x5
        layer1out = self.layer1(x)  # 3x3
        # spatial multi-scale feature fusion
        spatial_feature = torch.cat((layer1out, layer2out, layer3out), 1)
        spaitial_brach_out  = self.transconv3(spatial_feature)
        spaitial_brach_abun = self.encodelayer(spaitial_brach_out)
        spaitial_brach_res = self.decoderlayer5(spaitial_brach_abun)

        #spectral branch
        spectral_feature = self.spectral(x)
        spectral_brach_abun = self.encodelayer(spectral_feature)
        spectral_brach_res = self.decoderlayer6(spectral_brach_abun)


        #feature fusion of multi-scale spatial features and spectral features
        spectral_spatial_feature = torch.cat((x, spaitial_brach_out, spectral_feature), 1)

        #dimensionality reduction  --> 48
        spectral_spatial_feature2 = self.layer5(spectral_spatial_feature)

        # attention
        spectral_spatial_feature3 = self.conv(spectral_spatial_feature2)
        feature_attetion_weight = self.spectral_attention(spectral_spatial_feature3)
        spectral_attetion_feature = torch.mul((spectral_spatial_feature3 * feature_attetion_weight), 1)

        # shortcut
        attetion_feature = spectral_attetion_feature + spectral_spatial_feature2
        # channels-->endmember numbers
        layer1out = self.layer6(attetion_feature)
        layer1out = layer1out + self.layer7(attetion_feature)

        en_result = self.encodelayer(layer1out)
        de_result = self.decoderlayer4(en_result)
        return en_result, de_result, spaitial_brach_abun, spaitial_brach_res, spectral_brach_abun, spectral_brach_res