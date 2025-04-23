import subprocess
def aes(n):
    subprocess.run(['g++', 'aes.cpp', '-o', 'aes'])
    subprocess.run(['./aes', str(n)])
    
def ring(n):
    subprocess.run(['g++', 'ring.cpp', '-o', 'ring'])
    subprocess.run(['./ring', str(n)])
    
def mmm(n):
    subprocess.run(['g++', 'mmm.cpp', '-o', 'mmm'])
    subprocess.run(['./mmm', str(n)])
    
def sparse(n, m):
    subprocess.run(['g++', 'sparse.cpp', '-o', 'sparse'])
    subprocess.run(['./sparse', str(n), str(m)])
    
aes(3)
    
