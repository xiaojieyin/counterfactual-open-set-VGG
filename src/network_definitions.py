import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from vector import clamp_to_unit_sphere


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, scale=4, output_scale=4):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        if output_scale == 8:
            x = self.conv_out_6(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, scale * scale)
            return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        if output_scale == 4:
            x = self.conv_out_9(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, scale * scale)
            return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        if output_scale == 2:
            x = self.conv_out_10(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, scale * scale)
            return x

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = clamp_to_unit_sphere(x)
        return x


class generator(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512 * 2 * 2, bias=False)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        self.cuda()

    def forward(self, x, scale=4, input_scale=4):
        batch_size = x.shape[0]
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        # 3 x 32 x 32
        x = nn.Sigmoid()(x)
        return x


class multiclassDiscriminator(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, scale=4, **kwargs):
        super(self.__class__, self).__init__()
        self.scale = scale
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * self.scale * self.scale * 2, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x

        # Lazy minibatch discrimination: avg of other examples' features
        batch_avg = torch.exp(-x.mean(dim=0))
        batch_avg = batch_avg.expand(batch_size, -1)
        x = torch.cat([x, batch_avg], dim=1)
        x = self.fc1(x)
        return x


class _classifier(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, scale=4, **kwargs):
        super(self.__class__, self).__init__()
        self.scale = scale
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * self.scale * self.scale, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x
        x = self.fc1(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class encoder_vgg11(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.features = make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512])
        self.conv = nn.Conv2d(512, latent_size, 3, 1, 1, bias=False)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, scale=4, output_scale=4):
        batch_size = len(x)

        x = self.features(x)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = clamp_to_unit_sphere(x, scale * scale)
        return x


class generator_vgg(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512 * 2 * 2, bias=False)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        self.cuda()

    def forward(self, x, scale=4, input_scale=4):
        batch_size = x.shape[0]
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, scale, scale)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        x = self.conv6(x)
        # 3 x 32 x 32
        x = nn.Sigmoid()(x)
        return x


class multiclassDiscriminator_vgg(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, scale=4, **kwargs):
        super(self.__class__, self).__init__()
        self.scale = scale
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.features = make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512])
        self.conv = nn.Conv2d(512, 128, 1, 1, 0, bias=False)

        self.fc = nn.Linear(128 * self.scale * self.scale * 2, num_classes)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.features(x)
        x = self.conv(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x

        # Lazy minibatch discrimination: avg of other examples' features
        batch_avg = torch.exp(-x.mean(dim=0))
        batch_avg = batch_avg.expand(batch_size, -1)
        x = torch.cat([x, batch_avg], dim=1)
        x = self.fc(x)
        return x


class classifier_vgg11(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, scale=4, **kwargs):
        super(self.__class__, self).__init__()
        self.scale = scale // 2
        self.features = make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        # self.conv = nn.Conv2d(512, 128, 1, 1, 0, bias=False)
        #
        # self.fc = nn.Linear(128 * self.scale * self.scale, num_classes)
        self.fc = nn.Linear(512 * self.scale * self.scale, 4096)
        self.classifier = nn.Sequential(
            # nn.Linear(512 * self.scale * self.scale, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.features(x)
        # x = self.conv(x)

        x = x.view(batch_size, -1)
        # if return_features:
        #     return x
        x = self.fc(x)
        x = self.classifier(x)
        return x


def encoder_vgg(pretrained=False, **kwargs):
    model = encoder_vgg11(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Using pretrained weights for encoder")
    return model


def classifier_vgg(pretrained=False, **kwargs):
    model = classifier_vgg11(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Using pretrained weights for classifier")
    return model


def classifier(pretrained=False, **kwargs):
    model = _classifier(**kwargs)
    if pretrained:
        checkpoint = torch.load('/home/sdb2/yinxiaojie/osrci/model_best.pth.tar')
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if 'fc1' not in k}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Using pretrained weights for classifier")
    return model
