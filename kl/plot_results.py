import matplotlib.pylab as plt
import pickle

#kl_dict = {int(k):v for k,v in kl_dict.items()}
with open('kl_dict.pkl', 'rb') as f:
    kl_dict = pickle.load(f)

lists = sorted(kl_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x,y)
plt.title("Test set kl-divergence of TM & LM after the same amount \n of training steps")
plt.xlabel("# steps")
plt.ylabel("kl-divergence")
plt.savefig('kl.png')
plt.clf()
plt.xscale('log')
plt.plot(x, y)
plt.title("Test set kl-divergence of TM & LM after the same amount \n of training steps (lin-log)")
plt.xlabel("# steps")
plt.ylabel("kl-divergence")
plt.savefig('kl_log.png')
