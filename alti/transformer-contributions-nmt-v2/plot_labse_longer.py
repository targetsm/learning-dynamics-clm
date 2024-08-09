import pickle
import numpy
import sys
import matplotlib.pylab as plt

#fig, axs = plt.subplots(8, 2, sharex=False, constrained_layout=True)
# Remove horizontal space between axes
#fig.subplots_adjust(hspace=0.08)
epoch_dir = {'iwslt14deen/iwslt/': (1132, 5660), 
        'iwslt14deen/wmt/': (1132, 5660), 
        'wmt22deen_subset/iwslt/': (19564, 97820), 
        'wmt22deen_subset/wmt/': (19564, 97820),
        'wmt22frde_subset/iwslt/': (14632, 73160), 
        'wmt22frde_subset/wmt/': (14632, 73160), 
        'wmt22frde_subset/wmt_big/': (14632, 73160)} 
#dir_list_1 = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/']
#dir_list_2 = ['wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
#dir_list_lrp = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/', 'wmt22frde_subset/iwslt/', 'wmt22frde_subset/wmt/', 'wmt22frde_subset/wmt_big', 'wmt22frde/wmt/', 'wmt22frde/wmt_big/', 'wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
dir_list_3 = [ 'wmt22frde_subset/wmt/']
#dir_list_4 = ['wmt22frde/wmt/', 'wmt22frde/wmt_big/']
for k in range(3,4):
    fig, axs = plt.subplots(len(eval(f'dir_list_{k}'))*2, 2, sharex=False, constrained_layout=True)
    # Remove horizontal space between axes
    #fig.subplots_adjust(hspace=0.08)
    for i in range(0,len(eval(f'dir_list_{k}')), 1):
        with open(f'plt/tl/{eval(f"dir_list_{k}")[i]}/alti_results.pkl', 'rb') as f:
            alti_dict = pickle.load(f)
        lists = [(x, y) for x, y in sorted(alti_dict.items())] # sorted by key, return a list of tuples
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
        ax0.set_ylabel("contribution")
        #ax0.set_xlabel("# steps")
        ax0.set_yticks(numpy.arange(0, 1., 0.1))
        ax0.set_title(name+ ' ALTI+')
        
        if eval(f"dir_list_{k}")[i] in epoch_dir.keys():
            e_1, e_5 = epoch_dir[eval(f"dir_list_{k}")[i]]
            ax0.axvline(x=e_1, color='grey', ls='--')
            ax0.axvline(x=e_5, color='grey', ls='--')
    

        ax1 = axs[i*2][1]
        ax1.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
        ax1.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylabel("contribution")
        #ax1.set_xlabel("# steps")
        ax1.set_yticks(numpy.arange(0, 1., 0.1))
        ax1.set_title(name + ' ALTI+ (log)')
        ax1.set_xscale('log')
        ticks = 10**numpy.arange(2,6)
        ax1.set_xticks(ticks, ticks) 
    
        if eval(f"dir_list_{k}")[i] in epoch_dir.keys():
            e_1, e_5 = epoch_dir[eval(f"dir_list_{k}")[i]]
            ax1.axvline(x=e_1, color='grey', ls='--')
            ax1.axvline(x=e_5, color='grey', ls='--')


        with open(f'/local/home/ggabriel/ma/comp/hallucinations/labse/plt/{eval(f"dir_list_{k}")[i]}/labse.pkl', 'rb') as f:
            labse_dict = pickle.load(f)
    
        labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt)}
        lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
        x, y_l = zip(*lists) # unpack a list of pairs into two tuples
        y, y_v = zip(*y_l)
        ax2 = axs[i*2+1][0]
        ax2.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
        ax2.grid()
        ax2.set_ylabel("cos-sim")
        ax2.set_xlabel("# steps")
        ax2.legend()
        ax2.set_yticks(numpy.arange(0, 1., 0.1))
        ax2.update_from(axs[i*2+1][0])
        ax2.set_title(name+ ' LaBSE')
        
        if eval(f"dir_list_{k}")[i] in epoch_dir.keys():
            e_1, e_5 = epoch_dir[eval(f"dir_list_{k}")[i]]
            ax2.axvline(x=e_1, color='grey', ls='--')
            ax2.axvline(x=e_5, color='grey', ls='--')

    
        ax3 = axs[i*2+1][1]
        ax3.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
        ax3.grid()
        ax3.set_ylabel("cos-sim")
        ax3.set_xlabel("# steps")
        ax3.legend()
        
        ax3.set_yticks(numpy.arange(0, 1., 0.1))
        ax3.set_title(name+ ' LaBSE (log)')
        ax3.set_xscale('log')
        ticks = 10**numpy.arange(2,6)
        ax3.set_xticks(ticks, ticks) 
        if eval(f"dir_list_{k}")[i] in epoch_dir.keys():
            e_1, e_5 = epoch_dir[eval(f"dir_list_{k}")[i]]
            ax3.axvline(x=e_1, color='grey', ls='--')
            ax3.axvline(x=e_5, color='grey', ls='--')

    
    fig.set_size_inches(8,len(eval(f'dir_list_{k}'))*2*2)
    #fig.suptitle('ALTI+ mean source and target prefix contributions over the course of training', wrap=True)
    #plt.grid(True)
    #plt.xlabel("# steps")
    #plt.ylabel("contribution")
    #plt.yticks(numpy.arange(0, 1., 0.1))
    fig.tight_layout()
    plt.savefig(f'plt/alti_evolution_longer.pdf')
    
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
