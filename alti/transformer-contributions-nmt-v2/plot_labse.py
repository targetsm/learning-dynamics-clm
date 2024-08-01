import pickle
import numpy
import sys
import matplotlib.pylab as plt

#fig, axs = plt.subplots(8, 2, sharex=False, constrained_layout=True)
# Remove horizontal space between axes
#fig.subplots_adjust(hspace=0.08)

dir_list_1 = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/']
dir_list_2 = ['wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
#dir_list_lrp = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/', 'wmt22frde_subset/iwslt/', 'wmt22frde_subset/wmt/', 'wmt22frde_subset/wmt_big', 'wmt22frde/wmt/', 'wmt22frde/wmt_big/', 'wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
dir_list_3 = ['wmt22frde_subset/iwslt/', 'wmt22frde_subset/wmt/', 'wmt22frde_subset/wmt_big/']
dir_list_4 = ['wmt22frde/wmt/', 'wmt22frde/wmt_big/']
for k in range(1, 5):
    fig, axs = plt.subplots(len(eval(f'dir_list_{k}'))*2, 2, sharex=False, constrained_layout=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.08)
    for i in range(0,len(eval(f'dir_list_{k}')), 1):
        with open(f'plt/tl/{eval(f"dir_list_{k}")[i]}/alti_results.pkl', 'rb') as f:
            alti_dict = pickle.load(f)
        lists = [(x, y) for x, y in sorted(alti_dict.items()) if x <= 100000] # sorted by key, return a list of tuples
        print(eval(f'dir_list_{k}')[i]) 
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        #print(x)
        #print(y)
        src = [el['src'] for el in y]
        trg = [el['trg'] for el in y]
        #print(src)
        #y_src, y_tgt = zip(*y)
        src_mean = [numpy.mean(el) for el in src]
        src_se = [numpy.std(el)/numpy.sqrt(len(src)) for el in src]
        trg_mean = [numpy.mean(el) for el in trg]
        trg_se = [numpy.std(el)/numpy.sqrt(len(trg)) for el in trg]
        
        #plt.plot(x, y_src, linestyle='-', marker='.', label='source')
        #plt.plot(x, y_tgt, linestyle='-', marker='.', label='target prefix')
        name= eval(f'dir_list_{k}')[i].replace('/', '-').replace('_', '-').replace('wmt-big-', 'large').replace('wmt-', 'medium').replace('iwslt-', 'small')
        print(name)
        ax0 = axs[i*2][0]
        ax0.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
        ax0.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
        ax0.legend()
        ax0.grid(True)
        ax0.set_ylabel("kl-divergence")
        #ax0.set_xlabel("# steps")
        ax0.set_yticks(numpy.arange(0, 1., 0.1))
        ax0.set_title(name+ ' ALTI+')
        ax1 = axs[i*2][1]
        ax1.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
        ax1.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylabel("kl-divergence")
        #ax1.set_xlabel("# steps")
        ax1.set_yticks(numpy.arange(0, 1., 0.1))
        ax1.set_title(name + ' ALTI+ (log)')
        ax1.set_xscale('log')
    
        with open(f'/local/home/ggabriel/ma/comp/hallucinations/labse/plt/{eval(f"dir_list_{k}")[i]}/labse.pkl', 'rb') as f:
            labse_dict = pickle.load(f)
    
        labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt) <= 100000}
        lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
        x, y_l = zip(*lists) # unpack a list of pairs into two tuples
        y, y_v = zip(*y_l)
        ax0 = axs[i*2+1][0]
        ax0.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.')#, label=name + ' LaBSE cos_sim')
        ax0.grid()
        ax0.set_ylabel("cos-sim")
        ax0.set_xlabel("# steps")
        #ax0.legend()
        ax0.set_yticks(numpy.arange(0, 1., 0.1))
        ax0.update_from(axs[i*2+1][0])
        ax0.set_title(name+ ' LaBSE cos-sim')
    
    
        ax1 = axs[i*2+1][1]
        ax1.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.')
        ax1.grid()
        ax1.set_ylabel("cos-sim")
        ax1.set_xlabel("# steps")
        #ax1.legend()
        
        ax1.set_yticks(numpy.arange(0, 1., 0.1))
        ax1.set_title(name+ ' LaBSE cos-sim (log)')
        ax1.set_xscale('log')
    
    
    fig.set_size_inches(8,len(eval(f'dir_list_{k}'))*2*2)
    #fig.suptitle('ALTI+ mean source and target prefix contributions over the course of training', wrap=True)
    #plt.grid(True)
    #plt.xlabel("# steps")
    #plt.ylabel("contribution")
    #plt.yticks(numpy.arange(0, 1., 0.1))
    plt.savefig(f'plt/alti_evolution_{k}.pdf')
    
    #plt.xscale('log')
    #fig.suptitle('ALTI+ mean source and target prefix contributions over the course of training (lin-log)', wrap=True)
    #ticks = 10**numpy.arange(2,6)
    #plt.xticks(ticks, ticks)
    #plt.savefig(f'plt/alti_evolution_log_{k}.pdf')


#
#
#import pickle
#import numpy
#import sys
#import matplotlib.pylab as plt
#
#fig, axs = plt.subplots(8, 1, sharex=False, constrained_layout=True)
## Remove horizontal space between axes
#fig.subplots_adjust(hspace=0.08)
#
#dir_list_lrp = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/', 'wmt22frde_subset/iwslt/', 'wmt22frde_subset/wmt/', 'wmt22frde/wmt/', 'wmt22frde/wmt_big/', 'wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
#
#for i in range(len(dir_list)):
#    with open(f'/local/home/ggabriel/ma/comp/hallucinations/labse/plt/{dir_list[i]}/labse.pkl', 'rb') as f:
#        labse_dict = pickle.load(f)
#    labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt)}
#    lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
#    x, y_l = zip(*lists) # unpack a list of pairs into two tuples
#    y, y_v = zip(*y_l)
#
#    axs[i].errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
#    axs[i].grid()
#    axs[i].legend()
#    axs[i].set_yticks(numpy.arange(0, 1., 0.1))
#
#fig.suptitle('Evolution of mean LaBSE similarity during Training', wrap=True)
#plt.xlabel('# steps')
#plt.ylabel("Similarity")
#fig.set_size_inches(10,20)
#plt.savefig(f'plt/labse.pdf')
#
#plt.xscale('log')
#fig.suptitle('Evolution of mean LaBSE similarity during Training (lin-log)', wrap=True)
#ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
##ticks = [10000, 50000, 60000, 100000]
#plt.xticks(ticks, ticks)
#plt.savefig(f'plt/labse_log.pdf')
