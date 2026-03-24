import matplotlib
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def my_tsne(data,y):
    t_sne = TSNE(n_components=2)
    features = t_sne.fit_transform(data)
    matplotlib.rcParams.update(dict(
    zip(['legend.fontsize',
         'axes.labelsize',
         'axes.titlesize',
         'xtick.labelsize',
         'ytick.labelsize'],
        [16] * 5)
))
    candi_color = ['#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#002c53','#ffa510' ,'#0c84c6',
    '#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#002c53','#ffa510' ,'#0c84c6',
    '#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#002c53','#ffa510' ,'#0c84c6',
    '#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#002c53','#ffa510' ,'#0c84c6'
    ]

    fig=plt.figure(figsize=(6, 6))
    ax=plt.subplot(111)
    for i in range(len(set(y))):
        x0 = features[y == i, 0][:200]
        x1 = features[y == i, 1][:200]
        ax.scatter(x0, x1, c=candi_color[i], s=7)
    plt.xticks([])    # Remove x-axis ticks
    plt.yticks([])    # Remove y-axis ticks
    #plt.savefig('img.jpg',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.show()
 


    return

