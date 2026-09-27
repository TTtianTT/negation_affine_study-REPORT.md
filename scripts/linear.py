import numpy as np
from scipy.linalg import eigh,svd

class Design:
    def __init__(self,x):
        self.mean=x.mean(0); self.x=x-self.mean; e,q=eigh(self.x@self.x.T); self.e=np.maximum(e,0);self.q=q;self.scale=max(self.e.sum()/len(x),1e-12)
    def fit(self,target,alpha):
        mu=target.mean(0);d=target-mu; lam=alpha*self.scale
        qt=self.q.T@d; coef=self.q@(qt/(self.e[:,None]+lam))
        # singular vectors of augmented-design fitted response give exact rank-constrained ridge optimum
        z=np.sqrt(self.e/(self.e+lam))[:,None]*qt
        _,_,vt=svd(z,full_matrices=False,check_finite=False)
        return mu,coef,vt
    def predict(self,x,fit,rank=None,random_basis=None):
        mu,c,vt=fit; v=(x-self.mean)@self.x.T@c
        basis=random_basis if random_basis is not None else (vt[:rank].T if rank is not None else None)
        if basis is not None:v=(v@basis)@basis.T
        return v+mu
