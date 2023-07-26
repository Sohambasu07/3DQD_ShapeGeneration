import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_logs = pd.read_csv('../train_recon_loss.csv')
val_logs = pd.read_csv('../val_recon_loss.csv')
print(val_logs.keys())
train_loss = train_logs.loc[:, 'soft-shape-51 - Recon loss/Train']
val_loss = val_logs.loc[:, 'soft-shape-51 - Recon loss/Val']
'soft-shape-51 - Recon loss/Val'
down_idxs = np.arange(0, len(train_loss), 1000)

downsampled_train_loss = train_loss[down_idxs]
train_iters = np.arange(len(train_loss))[down_idxs]
fig = plt.figure()
plt.plot(train_iters, downsampled_train_loss, label='train')

val_iters = np.arange(len(val_loss)) * 2000
plt.plot(val_iters, val_loss, label='validation')
plt.yscale('log')
ticks = [1e-3, 3e-3, 1e-2, 1e-1]
plt.yticks(ticks, labels=ticks)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('MSE Reconstruction Loss')
plt.legend()
plt.grid()
fig.savefig('training_curves.png', dpi=1000)
plt.show()