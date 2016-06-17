from numpy import *

def sigmoid(x):
	return 1.0/(1.0 + exp(-x))

def softmax_simple(net, k):
	return exp(net[k])/sum([exp(net[j]) for j in range(len(net))])

def softmax(x):
	xt = exp(x - max(x))
	return xt / sum(xt)

def grad(x):
	return x*(1-x)

def make_onehot(i, n):
	y = zeros(n)
	y[i] = 1
	return y

def fraq_loss(vocab, word_to_num, vocabsize):
	fraction_lost = float(sum([vocab['count'][word] for word in vocab.index if (not word in word_to_num) and (not word == "UUUNKKK")]))
	fraction_lost /= sum([vocab['count'][word] for word in vocab.index if (not word == "UUUNKKK")])
	return fraction_lost

def adjust_loss(loss, funk, q, mode='basic'):
	if mode == 'basic':
		# remove freebies only: score if had no UUUNKKK
		return (loss + funk*log(funk))/(1 - funk)
	else:
		# remove freebies, replace with best prediction on remaining
		return loss + funk*log(funk) - funk*log(q)

class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = cumsum(p)

    def sample(self, k=1):
        rs = random.random(k)
        # binary search to get indices
        return searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p


def multinomial_sample(p):
    """
    Wrapper to generate a single sample,
    using the above class.
    """
    return MultinomialSampler(p).sample(1)[0]
