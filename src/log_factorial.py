def logfactorial(n):
    if n <= 1:
        return 0
    else:
        return (n*(np.log(n)-1)+0.5*np.log(2*np.pi*n))
