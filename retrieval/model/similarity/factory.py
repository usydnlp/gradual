# TODO: improve this
from . import similarity as sim
from addict import Dict


_similarities = {
    'cosine': {
        'class': sim.Cosine,
    },
    'adapt_t2i': {
        'class': sim.AdaptiveEmbeddingT2I,
    },
    'adapt_i2t': {
        'class': sim.AdaptiveEmbeddingI2T,
    },
    'scan_i2t': {
        'class': sim.StackedAttention,
    },
    'scan_t2i': {
        'class': sim.StackedAttention,
    },
    'order': None,
}

def get_similarity_object(name, q1_size, q2_size, v1_size, v2_size, **kwargs):
    settings = _similarities[name]
    return settings['class'](v1_size=v1_size, v2_size=v2_size, q1_size=q1_size, q2_size=q2_size, **kwargs)


def get_sim_names():
    return _similarities.keys()
