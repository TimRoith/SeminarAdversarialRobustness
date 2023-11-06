#%%
import torch
from helpers import load_model, read_image, show_image, classifier
import adv_attacks as adv
import matplotlib.pyplot as plt

#%%
model = load_model(name='ResNet50')
clf = classifier(model)
#%%
I = read_image('imgs/panda.jpg')
show_image(I)
clf(I, k=5)

#%% Attack
attack = adv.Linf_pgd(attack_iters=1, 
                      epsilon=1.1, 
                      alpha=0.0007,
                      init_mode = 'zeros',
                      verbosity = 1)
xadv = attack(model, I)

show_image(xadv)
clf(xadv, k=3)
