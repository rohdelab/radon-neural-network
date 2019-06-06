import numpy as np

import torch
from torch import optim



class GSW():
    def __init__(self,ftype='linear',nofprojections=10,degree=2,radius=2.,use_cuda=True):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW

    def gsw(self,X,Y):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if self.theta is None:
            self.theta=self.random_slice(dn)

        Xslices=self.get_slice(X,self.theta)
        Yslices=self.get_slice(Y,self.theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype=='poly':
            dpoly=self.homopoly(dn,self.degree)
            theta=torch.randn((1,dpoly),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype=='circular':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
            theta.data*=self.radius
        self.theta=theta
        
#         print(self.theta)
#             iterations=1000
#         else:
#             iterations=iterations
        optimizer=optim.Adam([self.theta],lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
#             print(X)
#             print(Y)
#             print(self.theta)
            loss=-self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.sqrt(torch.sum(self.theta.data**2))
#             print(loss.item())
#         plt.plot(-total_loss)
#         plt.show()
#         print(theta.data)
        return self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))

    def gsl2(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Yslices_sorted=torch.sort(Yslices,dim=0)

        return torch.sqrt(torch.sum((Xslices-Yslices)**2))

    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':            
            return self.poly(X,theta)
        elif self.ftype=='circular':
            return self.circular(X,theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([self.radius*th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        N,d=X.shape
        dhomopoly=self.homopoly(d,self.degree)
        

        if theta is None:
            dpoly=self.homopoly(d,self.degree)
            theta=torch.randn((1,dpoly),device=self.device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
            
        assert theta.shape[1]==dhomopoly

        oddlist=np.arange(self.degree+1)
        oddlist=oddlist[np.argwhere(np.mod(oddlist,2)==1)]

        HX=torch.ones((N,dhomopoly)).type(torch.FloatTensor).to(self.device)
        X=X.type(torch.FloatTensor).to(self.device)

        powers=list()
        for n,a in enumerate(oddlist):
            powers.append(list(self.get_powers(d,a.squeeze())))
        powers=np.concatenate(powers)

        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=(X[:,i]**float(p))

        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    
    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in the summation of homogeneous 
        polynomials of odd degrees< degree

        '''
        d=0
        oddlist=np.arange(degree+1)
        oddlist=oddlist[np.argwhere(np.mod(oddlist,2)==1)]

        for a in oddlist:
            d+=len(list(self.get_powers(dim,a.squeeze())))    
        return d

