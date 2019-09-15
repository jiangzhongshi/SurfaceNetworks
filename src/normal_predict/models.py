import torch
import torch.nn as nn
import sys
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils_pt as utils
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=6, out_features=1):
        super(Model, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)

        for i in range(15):
            if i % 2 == 0:
                module = utils.LapResNet2(128)
            else:
                module = utils.AvgResNet2(128)
            self.add_module('rn{}'.format(i), module)

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm='pre')

    def forward(self, L, mask, inputs):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = self.conv2(x)
        x = F.elu(x)

        return x


class LapDeepModel(nn.Module):
    def __init__(self, in_features=3, out_features=1, layers=15, bnmode='', nofirstId=False, only_lap = False, bottleneck=False, **useless):
        super(LapDeepModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm='')
        self.layer_num = layers
        self.firstL = None
        if nofirstId:
            self.firstL = utils.LapResNet2_noId(128, bnmode)
        if bottleneck:
            assert layers==16
            self.bottleneck = [128, 128, 64, 64, 32, 32, 16,16, 16, 16, 32, 32, 64,64,128,128, 128]
        else:
            self.bottleneck = [128]* (layers+1)
        for i in range(self.layer_num):
            if i % 2 == 0 or only_lap:
                module = _LapResNet2(self.bottleneck[i], self.bottleneck[i+1], bnmode)
            else:
                module = _AvgResNet2(self.bottleneck[i], self.bottleneck[i+1], bnmode)
            self.add_module('rn{}'.format(i), module)

        if bnmode is not None:
            bnmode += 'pre'
        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm=bnmode)

    def forward(self, L, mask, inputs, **kwargs):

        _, num_nodes, in_features = inputs.size()
        x = self.conv1(inputs)

        if self.firstL is not None:
            x = self.firstL(L, mask, x)
        for i in range(self.layer_num):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + repeating_expand(inputs, x.size(2))

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

class GatDeepModel(nn.Module):
    def __init__(self, in_features=3, out_features=1, layers=15, bnmode='', nofirstId=False, dense=True):
        super().__init__()

        assert dense
        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm='')
        self.layer_num = layers
        self.firstL = None
        if nofirstId:
            assert False
            self.firstL = utils.LapResNet2_noId(128, bnmode)
        for i in range(self.layer_num):
            if i % 2 == 0:
                module = pygat.GatResNet2(128, bnmode)
            else:
                module = utils.AvgResNet2(128, bnmode)
            self.add_module('rn{}'.format(i), module)

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm=bnmode+'pre')

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer_num):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)
        out_features = x.shape[2]
        if out_features <= 3:
            return x + inputs[:,:,:out_features]
        else:
            return x + inputs

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class AvgModel(nn.Module):

    def __init__(self, inp, out, layer):
        super(AvgModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(inp, 128, batch_norm=None)

        self.layer = layer
        for i in range(layer):
            module = utils.AvgResNet2(128)
            self.add_module('rn{}'.format(i), module)

        self.conv2 = utils.GraphConv1x1(128, out, batch_norm='pre')

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class MlpModel(nn.Module):

    def __init__(self, inp, out, layer):
        super(MlpModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(inp, 128, batch_norm=None)

        self.layer = layer
        for i in range(layer):
            module = utils.MlpResNet2(128)
            self.add_module('rn{}'.format(i), module)

        self.bn = utils.GraphBatchNorm(128)
        self.conv2 = utils.GraphConv1x1(128, out, batch_norm=None)

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = self.bn(x)
        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

class DirModel(nn.Module):
    def __init__(self, in_features = 3, out_features = 1):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)

        for i in range(15):
            if i % 2 == 0:
                module = utils.DirResNet2(128)
            else:
                module = utils.AvgResNet2(128)

            self.add_module('rn{}'.format(i), module)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm='pre')


    def forward(self, Di, DiA, mask, inputs):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(2) // 4

        f = v.new_zeros(batch_size, num_faces, 128)

        for i in range(15):
            if i % 2 == 0:
                v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
            else:
                v = self._modules['rn{}'.format(i)](None, mask, v)

        x = v
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        return x

class DirDeepModel(nn.Module):

    def __init__(self, in_features = 3, out_features = 1, layers = 30):
        super(DirDeepModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)
        self.feature_width = 128
        self.layer_num = layers
        for i in range(self.layer_num):
            if i % 2 == 0:
                module = utils.DirResNet2(128)
            else:
                module = utils.AvgResNet2(128)

            self.add_module('rn{}'.format(i), module)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm='pre')


    def forward(self, DiDA, mask, inputs):
        Di, DiA = DiDA
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(-1) // 4 //batch_size

        f = v.new_zeros(size=(batch_size, num_faces, self.feature_width))

        for i in range(self.layer_num):
            if i % 2 == 0:
                v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
            else:
                v = self._modules['rn{}'.format(i)](None, mask, v)

        x = v
        x = self.conv2(x)
        x = F.elu(x)
        return x

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

class DirModelToFace(nn.Module):
    def __init__(self, in_features = 3, out_features = 1):
        super(DirModelToFace, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)

        for i in range(16):
            if i % 2 == 0:
                module = utils.DirResNet2(128)
            else:
                module = utils.AvgResNet2(128)

            self.add_module('rn{}'.format(i), module)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm='pre')


    def forward(self, DiDA, mask, inputs):
        Di, DiA = DiDA
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 128))
        if v.is_cuda:
            f = f.cuda()

        for i in range(16):
            if i % 2 == 0:
                v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
            else:
                v = self._modules['rn{}'.format(i)](None, mask, v)

        x = f
        x = F.elu(x)
        x = self.conv2(x)
        return x

class IdResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(IdResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm='pre')
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm='pre')

    def forward(self, L, mask, inputs):
        x = inputs
        x = F.elu(x)

        xs = [x, x]#, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)

        x = F.elu(x)
        xs = [x, x]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs



class IdDeepModel(nn.Module):
    def __init__(self, in_features=3, out_features=1, layers=15):
        super(IdDeepModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)
        self.layer_num = layers
        for i in range(self.layer_num):
            module = IdResNet2(128)
            self.add_module('rn{}'.format(i), module)

        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm='pre')

    def forward(self, L, mask, inputs):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer_num):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = self.conv2(x)
        x = F.elu(x)

        return x

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)



class LapMATModel(nn.Module):
    '''
    Laplacian Network with Medial Axis Transform and double supervision
    '''
    def __init__(self, in_features=3, out_features=2, layers=15):
        super().__init__()
        assert out_features % 2 == 0
        self.LapModel = LapDeepModel(in_features, int(out_features/2), layers)

    def ma_output(self, L, mass, x):
        x = F.elu(x)
        sqmass = torch.sqrt(mass)
        spm = SPB1MM(L)
        x = spm(x)
        x = sqmass * x
        return torch.clamp(x, -4, 4)

    def forward(self, op, mask, inputs):
        L, mass = op
        outputs = self.LapModel(L, mask, inputs)
        ma = self.ma_output(L, mass, outputs)
        x = torch.cat([outputs, ma], dim = 2)
        return x

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


'''
Below are implementations of Cascade with the efficient pooling in Deff2017
'''


class _AvgResNet2(nn.Module):
    def __init__(self, num_inputs, num_outputs=None, bnmode='', inner_layers = 2):
        super().__init__()
        if num_outputs is None:
            num_outputs = num_inputs
        self.num_outputs = num_outputs

        if bnmode is not None:
            bnmode = bnmode + 'pre'
        self.add_module(f'bn_fc{0}', utils.GraphConv1x1(2 * num_inputs, num_outputs, batch_norm=bnmode))
        self.layer = inner_layers
        for i in range(1, self.layer):
            bn_fc = utils.GraphConv1x1(2 * num_outputs, num_outputs, batch_norm=bnmode)
            self.add_module(f'bn_fc{i}', bn_fc)

    def forward(self, L, mask, inputs):
        x = inputs
        for i in range(self.layer):
            x = F.elu(x)
            xs = [x, utils.global_average(x, mask).expand_as(x).contiguous()]
            x = torch.cat(xs, 2)
            x = self._modules[f'bn_fc{i}'](x)

        if self.num_outputs <= inputs.size(2):
            return x + inputs[:,:,:self.num_outputs]
        else:
            return x + torch.cat([inputs]*2, dim=2)


class _LapResNet2(nn.Module):
    def __init__(self, num_inputs, num_outputs=None, bnmode='', inner_layers = 2):
        super().__init__()
        if num_outputs is None:
            num_outputs = num_inputs
        self.num_outputs = num_outputs

        if bnmode is not None:
            bnmode = bnmode + 'pre'
        self.add_module(f'bn_fc{0}', utils.GraphConv1x1(2 * num_inputs, num_outputs, batch_norm=bnmode))
        self.layer = inner_layers
        for i in range(1, self.layer):
            bn_fc = utils.GraphConv1x1(2 * num_outputs, num_outputs, batch_norm=bnmode)
            self.add_module(f'bn_fc{i}', bn_fc)

    def forward(self, L, mask, inputs):
        x = inputs
        for i in range(self.layer):
            x = F.elu(x)
            if (L.layout) is torch.strided:
                xs = [x, torch.bmm(L,x)]
            else:
                batch, node, feat = x.size()
                xs = [x, torch.mm(L,x.view(-1, feat)).view(batch, node, feat)]
            x = torch.cat(xs, 2)
            x = self._modules[f'bn_fc{i}'](x)

        if self.num_outputs <= inputs.size(2):
            return x + inputs[:,:,:self.num_outputs]
        else:
            return x + torch.cat([inputs]*2, dim=2)

## Layers for Cascade Model
class _DeffLapDownResNet2(nn.Module):
    def __init__(self, num_outputs, inner_layers, bnmode, with_avg):
        super().__init__()
        self.lap0 = _LapResNet2(num_outputs, bnmode = bnmode, inner_layers = inner_layers)
        self.avg0 = lambda L, m, x:x
        if with_avg:
            self.avg0 = utils.AvgResNet2(num_outputs, bnmode = bnmode)

    def forward(self, L, mask, input):
        x = self.lap0(L, mask, input)
        x = self.avg0(L, mask, x)
        return x

class _DeffLapUpResNet2(nn.Module):
    def __init__(self, num_outputs, inner_layers, bnmode, with_avg):
        super().__init__()
        self.lap0 = _LapResNet2(num_outputs, num_outputs, bnmode=  bnmode, inner_layers = inner_layers)
        self.avg0 = lambda L, m, x:x
        if with_avg:
            self.avg0 = utils.AvgResNet2(num_outputs, bnmode = bnmode)

    def forward(self, L, mask, input_x):
        ''' Upsample the input and combine with down, then lapresnet
        Laps[k] : (b by V_k by V_k)
        input_x : (b by V_{k-1} by d)
        down_x: (b by V_k by d)

        Output: (b by V_k by d)
        '''
        _, in_nodes, num_features = input_x.size()

        x = input_x
        x = self.lap0(L, None, x)
        x = self.avg0(L, mask, x)

        return x

class _LaplacianPooling(nn.Module):
    def __init__(self, num_inputs, down = True):
        super().__init__()
        if down:
            self.num_outputs = num_inputs // 2
        else:
            self.num_outputs = num_inputs * 2
        self.num_inputs = num_inputs
        self.lap =  _LapResNet2(num_inputs, self.num_outputs, inner_layers = 1, bnmode = '')
    def forward(self, L, x):
        return self.lap(L, None, x).view(x.size(0), -1, self.num_inputs)

class EfficientCascade(nn.Module):

    def __init__(self, in_features=3, out_features=3, cascade_levels=4,
                 inner_layers = 2, bnmode='', with_avg=False, naive_pool=True, bottleneck=False, **kwargs):
        super().__init__()

        self.conv1 = utils.GraphConv1x1(in_features, 128, batch_norm=None)
        self.cascade_num = cascade_levels
        self.pools = {}
        if bottleneck:
            self.bottleneck = [16, 32, 64, 128]
        else:
            self.bottleneck = [128]*cascade_levels
        for i in range(self.cascade_num-1,0,-1): # 3,2,1
            module = _LapResNet2(num_inputs=self.bottleneck[i], num_outputs = self.bottleneck[i-1], inner_layers = inner_layers, bnmode = bnmode)
            self.add_module(f'down_rn{i}', module)
            if naive_pool:
                module = self._Pooling(down=True)
            else:
                module = _LaplacianPooling(self.bottleneck[i-1], down=True)
            self.add_module(f'down_pool{i}', module)

        self.lap0 = _LapResNet2(self.bottleneck[0], inner_layers=inner_layers) # 0 -> 0'

        for i in range(1, self.cascade_num): # 1',2',3'
            module = _LapResNet2(num_inputs = self.bottleneck[i-1],
                                 num_outputs = self.bottleneck[i],
                                inner_layers= inner_layers, bnmode = bnmode)
            self.add_module(f'up_rn{i}', module)
            if naive_pool:
                module = self._Pooling(down=False)
            else:
                module = _LaplacianPooling(self.bottleneck[i], down=False)
            self.add_module(f'up_pool{i}', module)

        if bnmode is not None:
            bnmode += 'pre'
        self.conv2 = utils.GraphConv1x1(128, out_features, batch_norm=bnmode)

    class _Pooling(nn.Module):
        def __init__(self,down=True):
            super().__init__()
            if down:
                self.pool = torch.nn.MaxPool1d(2, stride=2)
            else:
                self.pool = lambda x: F.interpolate(x, scale_factor=2)
        def forward(self, L, x):
            return self.pool(x.transpose(1,2)).transpose(1,2)

    def forward(self, Laps, mask, inputs, **kwargs):

        _, num_nodes, in_features = inputs.size()
        x = self.conv1(inputs) # V3

        down_series = []
        mask_series = []
        ma = mask
        for i in range(self.cascade_num-1, 0, -1): # 3,2,1
            down_series.append(x)
            mask_series.append(ma)
            x = self._modules['down_rn{}'.format(i)](Laps[i], ma, x)
            x = self._modules[f'down_pool{i}'](Laps[i], x)
            ma = torch.nn.MaxPool1d(2, stride=2)(ma.transpose(1,2)).transpose(1,2) # due to the specific construction of permuted nodes, this is valid: 0, 1 -> 1

        x = self.lap0(Laps[0], None, x) # V0

        for i in range(1, self.cascade_num): # 1,2,3
            x = self._modules[f'up_pool{i}'](Laps[i-1], x)
            x += down_series[-i][:,:,:x.size(2)]
            x = self._modules['up_rn{}'.format(i)](Laps[i], mask_series[-i], x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + repeating_expand(inputs, x.size(2))

    def fuzzy_load(self, pre_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def repeating_expand(inputs, out_features):
    _, _, in_features = inputs.size()
    times = out_features // in_features
    remin = out_features % in_features
    expanded_input = torch.cat([inputs]*times + [inputs[:,:,:remin]],dim=2)
    return expanded_input

class GlobalLocalModel(nn.Module):
    def __init__(self, in_features=3, out_features = 1,
                global_opts={}, local_opts = {}, **kwargs):
        super().__init__()
        self.global_net = EfficientCascade(in_features, out_features + 1, **global_opts) # + weight
        self.local_net = LapDeepModel(in_features, out_features, **local_opts)

    def forward(self, laplacian, mask,inputs, sigmoid=False, debug=False, **kwargs):
        score_weight_global = self.global_net(laplacian[0], mask[0], inputs)
        score_local = self.local_net(laplacian[1], mask[1], inputs)
        score_global = score_weight_global[:,:,:1]
        weight_global = score_weight_global[:,:,:-1]
        weight_global = torch.sigmoid(weight_global)

        if sigmoid:
            score_global = torch.sigmoid(score_global)
            score_local = torch.sigmoid(score_local)
        score_final = weight_global * (score_global) + (1 - weight_global) * (score_local)

        if debug: return score_global, score_local, score_final, weight_global # optional debugging output
        return torch.cat([score_global, score_local,score_final], dim = 1)

    def fuzzy_load(self, global_pre_dict, local_pre_dict):
        self.global_net.fuzzy_load(globl_pre_dict)
        self.local_net.fuzzy_load(local_pre_dict)
