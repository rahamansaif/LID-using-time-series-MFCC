def divAud(y, fs):
    import numpy as np
    import librosa as lib
    framesz = int(fs*5);
    #also does silence removal below a db value
    #interval = lib.effects.split(y, top_db = 30, frame_length = framesz, hop_length = hop);
    nframes = int(len(y)/framesz);
    y = np.array([y]);
    if(nframes > 1):
        y = y[:, 0:(nframes*framesz)];
        y.shape = (nframes, framesz);
    return y;