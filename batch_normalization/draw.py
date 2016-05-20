import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np

mlp_sgd_acc_train=np.loadtxt("data/mlp_sgd_acc_train.txt")
mlp_sgd_acc_val=np.loadtxt("data/mlp_sgd_acc_val.txt")
mlp_sgd_loss_train=np.loadtxt("data/mlp_sgd_loss_train.txt")
mlp_sgd_loss_val=np.loadtxt("data/mlp_sgd_loss_val.txt")

mlpbn_sgd_acc_train=np.loadtxt("data/mlpbn_sgd_acc_train.txt")
mlpbn_sgd_acc_val=np.loadtxt("data/mlpbn_sgd_acc_val.txt")
mlpbn_sgd_loss_train=np.loadtxt("data/mlpbn_sgd_loss_train.txt")
mlpbn_sgd_loss_val=np.loadtxt("data/mlpbn_sgd_loss_val.txt")

mlp_svrg_acc_train=np.loadtxt("data_svrg/_mlpbnFalse_SVRG_acc_train.txt")
mlp_svrg_acc_val=np.loadtxt("data_svrg/_mlpbnFalse_SVRG_acc_val.txt")
mlp_svrg_loss_train=np.loadtxt("data_svrg/_mlpbnFalse_SVRG_loss_train.txt")
mlp_svrg_loss_val=np.loadtxt("data_svrg/_mlpbnFalse_SVRG_loss_val.txt")

mlpbn_svrg_acc_train=np.loadtxt("data_svrg/_mlpbnTrue_SVRG_acc_train.txt")
mlpbn_svrg_acc_val=np.loadtxt("data_svrg/_mlpbnTrue_SVRG_acc_val.txt")
mlpbn_svrg_loss_train=np.loadtxt("data_svrg/_mlpbnTrue_SVRG_loss_train.txt")
mlpbn_svrg_loss_val=np.loadtxt("data_svrg/_mlpbnTrue_SVRG_loss_val.txt")

count = np.arange(mlp_sgd_acc_train.shape[0])+1

# print mlp_sgd_acc_train

#PLOT 
matplotlib.rcParams.update({'font.size': 16})
plt.figure(1)
plt.title('Loss of Validation Set')
plt.plot(count, mlp_sgd_loss_val, 'bo-',label="MLP SGD", linewidth='1')
plt.plot(count, mlpbn_sgd_loss_val, 'b^--',label="MLPBN SGD", linewidth='1')
plt.plot(count, mlp_svrg_loss_val, 'rs:',label="MLP SVRG", linewidth='1')
plt.plot(count, mlpbn_svrg_loss_val, 'r*-.',label="MLPBN SVRG", linewidth='1')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
pylab.savefig('figure/'+'CrossModel_Loss of Validation Set'+'.png',
   bbox_inches='tight')

plt.figure(2)
plt.title('Predict Accuracy of Validation Set')
plt.plot(count, mlp_sgd_acc_val, 'bo-',label="MLP SGD")
plt.plot(count, mlpbn_sgd_acc_val, 'b^--',label="MLPBN SGD")
plt.plot(count, mlp_svrg_acc_val, 'rs:',label="MLP SVRG", linewidth='1')
plt.plot(count, mlpbn_svrg_acc_val, 'r*-.',label="MLPBN SVRG", linewidth='1')
plt.xlabel('# Epochs')
plt.ylabel('Predict Accuracy')
plt.legend(bbox_to_anchor=(1,0.25))
# plt.show()
pylab.savefig('figure/'+'CrossModel_Predict Accuracy of Validation Set'+'.png',
   bbox_inches='tight')

plt.figure(3)
plt.title('Loss of Training Set')
plt.plot(count, mlp_sgd_loss_train, 'bo-',label="MLP SGD")
plt.plot(count, mlpbn_sgd_loss_train, 'b^--',label="MLPBN SGD")
plt.plot(count, mlp_svrg_loss_train, 'rs:',label="MLP SVRG", linewidth='1')
plt.plot(count, mlpbn_svrg_loss_train, 'r*-.',label="MLPBN SVRG", linewidth='1')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
pylab.savefig('figure/'+'CrossModel_Loss of Training Set'+'.png',
   bbox_inches='tight')

plt.figure(4)
plt.title('Predict Accuracy of Training Set')
plt.plot(count, mlp_sgd_acc_train, 'bo-',label="MLP SGD")
plt.plot(count, mlpbn_sgd_acc_train, 'b^--',label="MLPBN SGD")
plt.plot(count, mlp_svrg_acc_train, 'rs:',label="MLP SVRG", linewidth='1')
plt.plot(count, mlpbn_svrg_acc_train, 'r*-.',label="MLPBN SVRG", linewidth='1')
plt.xlabel('# Epochs')
plt.ylabel('Predict Accuracy')
plt.legend(bbox_to_anchor=(1,0.25))
# plt.show()
pylab.savefig('figure/'+'CrossModel_Predict Accuracy of Training Set'+'.png',
   bbox_inches='tight')

