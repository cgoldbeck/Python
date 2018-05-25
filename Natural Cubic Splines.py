# -*- coding: utf-8 -*-
# PURPOSE: Implements Natural Cubic Splines, Algorithm 3.4 in Faires Numerical Analsysis
#
#         To construct the cubic spline interpolant S for the function f:
#
# INPUT: numbers x = [x0, x1, ..., xn]; values f = [f(x0), f(x1), ..., f(xn)] as 
#        F[0, 0], F[1, 0], ..., F[n, 0]. 
#
# OUTPUT: the numbers aj , bj , cj , dj where
#
#        Sj = Sj(x) = aj + bj(x − xj) + cj(x - xj)^2 + dj(x − xj)^3,
#                        for j = 0, 1, . . . , n − 1.
#
#        
#==============================================================
# Record of revisions:
#
#   Date         Programmer         Description of change
#   ========   =================   ===================
#   11/14   Cameron Goldbeck      Original code
#   05/18   Cameron Goldbeck      Updated for Git
#
#==============================================================
import numpy

def naturalcs(x,f):
    n=len(x)-1
    #Step 0 : set h, distance between x points
    h=[0]*n
    for i in range(n):
        h[i]=x[i+1]-x[i]
        
    #Step 1 : define a in the system Ax=a which we want to solve
    a=[0]*(n)
    for i in range(1,n):
        a[i]=(3*(h[i]**(-1)))*(f[i+1]-f[i])-(3*(h[i-1]**(-1)))*(f[i]-f[i-1])
    
    #Step 2 : intialize coponents to solve tridiagonal matrix A
    l=[1]*(n+1)
    m=[0]*(n+1)
    z=[0]*(n+1)
    
    #Step 3 : solve tridiagonal systme
    for i in range(1,n):
        l[i]=2*(x[i+1]-x[i-1])-h[i-1]*m[i-1]
        m[i]=h[i]*(l[i]**(-1))
        z[i]=(a[i]-h[i-1]*z[i-1])*(l[i]**(-1))
    
    #Step 4 : construct coefficients to natural cubic splines
    c=[0]*(n+1)
    b=[0]*n
    d=[0]*n
    for j in range(1,n+1):
        k=n-j
        c[k]=z[k]-m[k]*c[k+1]
        b[k]=(f[k+1]-f[k])*(h[k]**(-1))-h[k]*(c[k+1]+2*c[k])*.33333333
        d[k]=(c[k+1]-c[k])*((3*h[k])**(-1))
        
    #Step 5 : make matrix of solutions
    F=numpy.zeros((n,4))
    for i in range(n):
        F[i,0]=f[i]
        F[i,1]=b[i]
        F[i,2]=c[i]
        F[i,3]=d[i]
        
    return F