from .unet import Unet as FC_EF
from .siamunet_conc import SiamUnet_conc as FC_Siam_conc
from .siamunet_diff import SiamUnet_diff as FC_Siam_diff

# 这样你在外面只需要：
# from models.comparison.FC_EF_Source import FC_EF, FC_Siam_conc, FC_Siam_diff