import torch 
import torch.nn as nn

from model.detr.util.misc import NestedTensor

class PrositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """