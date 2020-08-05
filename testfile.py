
from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet
import time
mol = Molecule.from_name('Be')
net = PauliNet.from_hf(mol).cuda()
net.use_sloglindet = 'never'
net.use_vandermonde = True

#%%

start_time = time.time()
print('running')
train(net, batch_size=100, n_steps=500)
end_time = time.time()

#%%

evaluate(net)

#%%



