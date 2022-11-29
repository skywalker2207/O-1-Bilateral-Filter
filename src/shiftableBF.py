import numpy as np
import math 
from math import sqrt

def gaussian_Kernel(size, sigma=1, twoDimensional=True):
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)

def applyFilter(img, filter):
  # Padding image
  maxVal = max(filter.shape)
  img3 = np.zeros((img.shape[0] + 2 * maxVal, img.shape[1] + 2 * maxVal))
  img3 = img3.astype('complex')
  img3[maxVal:img3.shape[0] - maxVal, maxVal:img3.shape[1] - maxVal] = img
  filter = filter.astype('complex')
  img2 = img3.copy()
  # Convolution
  for r in range(filter.shape[0], img2.shape[0] - filter.shape[0] + 1):
    for c in range(filter.shape[1], img2.shape[1] - filter.shape[1] + 1):
      img2[r][c] = np.sum(img3[r - int(filter.shape[0] / 2):r - int(filter.shape[0] / 2) + filter.shape[0], c - int(filter.shape[1] / 2):c - int(filter.shape[1] / 2) + filter.shape[1]] * filter)
  # Remove padded region and return image
  return img2[maxVal:img2.shape[0] - maxVal, maxVal:img2.shape[1] - maxVal]

def max_local_range ( fin  , w ) :

    
        T = -1 
        sym = ( w - 1 ) / 2 
        m, n = np.shape ( fin ) 
        template = fin.copy( )
        # scan along row
        for ii in range ( m ):
                L = np.zeros ( n )
                R = L.copy()
                L[ 0 ] , R[n-1] = template[ii,  0 ] , template[ii, n - 1 ]
                for k in range ( 1 , n):
                        if  (k % w) ==  0 :
                                L[k] = template[ii , k]
                                R[(n-1) - k] = template[ii , (n-1) - k ] # index in R corresponds to first element when k is in the last iteration
                        else :
                                L[k] = max ( L[k - 1 ], template[ii, k])
                                R[(n-1) - k ] = max ( R[n - k ], template[ii, (n -1) - k ])
                for k in range ( n):
                        
                        p , q = int(k - sym) , int(k + sym)
                        r = R[p] if p >=  0  else R[-1]
                        l = L[q] if q < n else L[-1]

                        template[ii, k] = max (r,l)

        # scan along column
        for jj in range ( n ):
                L = np.zeros ( m )
                R = L.copy ( )
                L[ 0 ] , R[m-1] = template[ 0 , jj] ,  template[m - 1 , jj]
                for k in range ( 1 , m):
                        if  (k  % w) ==  0 :
                                L[k] = template[k, jj]
                                R[(m-1) - k ] = template[(m-1) - k , jj]
                        else :
                                L[k] = max ( L[k - 1 ], template[k, jj])
                                R[(m-1)-k ] = max ( R[m - k ], template[(m -1)-k, jj])
                for k in range ( m):
                        p , q = int(k - sym) , int ( k + sym)
                        r = R[p] if p >=  0  else R[-1]
                        l = L[q] if q < m else L[-1]
                                
                        if k < m:
                                temp = max (r,l) - fin[k, jj]
                                T = max (T, temp)
       
        return int ( T )
        
        
def nCr(n,r):
    return math.comb ( int(n), int(r) )
    # f = math.factorial
    # return f(n) / f(r) / f(n-r)
    
def logfactorial(n):
    if n <= 1:
        return 0
    else:
        return (n*(np.log(n)-1)+0.5*np.log(2*np.pi*n))
    
def shiftableBF(f0,sigmas,filt,sigmar,w,tol):
    # f0 is the input image
    # sigmas is the standard deviation of the spatial Gaussian
    # sigmar is the standard deviation of the range Gaussian
    # w is the window size
    # tol is the tolerance for the stopping criteria
    # f is the output image
    # t is the time taken for the algorithm to run
    
    
    # Get image size
    m,n  = np.shape ( f0)
    
    # filt = cv2.getGaussianKernel((w,w), sigmas)
    filt = gaussian_Kernel(w,sigmas, twoDimensional= True )
    
    # im_filtered = applyFilter(f0,filt)
    
    T = max_local_range(f0,w)
    print ( "T = " , T )
    N = np.ceil(0.405*((T/sigmar)**2)).astype ( float)
    print("N = " , N)
    
    gamma = 1/(sqrt(N)*sigmar)
    twoN = math.pow(2,N)
    print(twoN)

    N = int ( N )
    if (tol==0):
        M=0
    else:
        if(sigmar>40):
             M = 0
        elif (sigmar > 10):
             sumCoeffs = 0
             for k in range(0,(N+3)//2):
                sumCoeffs +=  nCr(N,k)/twoN
                if (sumCoeffs > (tol/2)):
                    M = k
                    break
        else:
            M = np.ceil(0.5*(N - sqrt(4*N*np.log(2/tol))))
    
    # Initialize output image
    f = np.zeros((m,n))
    fnum = np.zeros((m,n))
    fdenom = np.zeros((m,n))
    # ii = sqrt(-1)
    ii = 1j
    
    M = int(M)
    if(N<50):
        for k in range(M,1+N-M):
            omegak = (2*k - N)*gamma
            bk = nCr(N,k) / twoN
            H  = np.exp(-ii*omegak*f0)
            G  = np.conj(H)
  
            F  = G*f0 
            barF = applyFilter(F,filt)
            barG = applyFilter(G,filt)
           
            
            fnum =  fnum + bk * H * barF
            fdenom  = fdenom + bk * H * barG
            
    else:
        for k in range(N,1+N-M):
            omegak = (2*k - N)*gamma
            # use Sterling's approximation
            bk = math.exp(logfactorial(N) - logfactorial(k)- logfactorial(N-k) - N*np.log(2))
            H  = math.exp(-ii*omegak*f0)
            G  = np.conj(H)
            F  = G*f0
            barF  = applyFilter(F,filt)
            barG = applyFilter(G, filt)
            fnum =  fnum + bk * H * barF
            fdenom  = fdenom + bk * H * barG
   
   
    idx1 = np.argwhere( fdenom < 1e-3)
    idx2 = np.argwhere( fdenom > 1e-3)
    p = idx1.shape[0]
    
    for i in range(0,p):
        f[idx1[i,0],idx1[i,1]] = f0[idx1[i,0],idx1[i,1]]
        
    q = idx2.shape[0]
    for i in range(0,q):
        f[idx2[i,0],idx2[i,1]] = np.real(fnum[idx2[i,0],idx2[i,1]]/fdenom[idx2[i,0],idx2[i,1]])
   
    
   
    
    return f,T,N,M

