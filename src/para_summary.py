from model.ssl import EDSR_Zoom
from option import args
model = EDSR_Zoom(args)

from model.edsr import EDSR
model = EDSR(args)
#
from model.rdn import RDN
model = RDN(args)
#
# from model.san import SAN
# model = SAN(args)
#
# from model.rcan import RCAN
# model = RCAN(args)

model = model.cuda()
from torchsummary import summary
channels = 3
H = 32
W = 32
summary(model, input_size=(channels, H, W))
