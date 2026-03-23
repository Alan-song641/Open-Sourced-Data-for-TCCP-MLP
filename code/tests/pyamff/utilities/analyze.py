
import matplotlib.pyplot as plt 
import os

def plotGradFlow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            print(" gradient:",p.grad)
    #plt.figure(figsize=(20,20))
#    x0, x1, y0, y1 = plt.axis()
#    margin = 0.25
#    plt.axis((x0 - margin,
#              x1 + margin,
#              y0 - margin,
#              y1 + margin))
    #plt.subplots_adjust(left=-1, bottom=-1, right=1, top=1, wspace=0, hspace=0)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.margins(0.2)
    plt.savefig(os.getcwd()+'/grad.png')


def truncatedNormal(mean=0.0, stddev=1.0, m=1):
    '''
    The generated values follow a normal distribution with specified 
    mean and standard deviation, except that values whose magnitude is 
    more than 2 standard deviations from the mean are dropped and 
    re-picked. Returns a vector of length m
    '''
    samples = []
    for i in range(m):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == m, "something wrong"
    if m == 1:
        return samples[0]
    else:
        return np.array(samples)
