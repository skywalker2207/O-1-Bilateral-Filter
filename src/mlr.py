import numpy as np 

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
        return T