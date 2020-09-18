from fastai.vision.all import *
from fastai.vision.widgets import *

def get_x(r):
    return path / 'all' / r['fname']
    
def get_y(r):
    return r['label'].split(' ')

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

path = Path()
learn_inf = load_learner(path/'flowers.pkl', cpu=True)

def predict(pic):
    predcls, calls, probs = learn_inf.predict(pic)
    t = 0.9  # empirically, threshold of 0.9 seems best
    cts = np.array(list(learn_inf.dls.vocab))
    idx = np.array((probs > t).nonzero().squeeze(-1))
    labels, probabilities = list(cts[idx]), probs[idx].tolist()
    return labels, probabilities