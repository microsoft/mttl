from enum import Enum


def convert_hps_to_dict(hparams):
    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        v = v.name if isinstance(v, Enum) else v
        hparams_allowed[k] = v
    return hparams_allowed
