import numpy as np

def crop_and_concat(encoder_features: np.ndarray,
                    decoder_features: np.ndarray) -> np.ndarray:
    """
    Returns the centered encoder crop concatenated with decoder features.
    """
    enc = np.asarray(encoder_features, dtype=np.float64)
    dec = np.asarray(decoder_features, dtype=np.float64)

    _, h_dec, w_dec, _ = dec.shape
    _, h_enc, w_enc, _ = enc.shape

    # crop position
    top = (h_enc - h_dec) // 2
    left = (w_enc - w_dec) // 2

    # center crop
    enc_cropped = enc[:, top : top + h_dec, left : left + w_dec, :]

    # concat: encoder_features first, then decoder_features
    out = np.concatenate([enc_cropped, dec], axis=-1)

    return out