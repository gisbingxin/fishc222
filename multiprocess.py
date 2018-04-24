import multiprocessing as mp
import threading as td
import time
def job(q):
     res=0
     for i in range(1000000):
       res += i + i**3+ i**2
     q.put(res)

def multipro():
    q=mp.Queue()
    p1=mp.Process(target=job, args=(q,))
    p2=mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1=q.get()
    res2=q.get()
    print('multipro:',res1 + res2)
def multithread():
    q=mp.Queue()
    t1=td.Thread(target=job, args=(q,))
    t2=td.Thread(target=job, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1=q.get()
    res2=q.get()
    print('multithread:',res1 + res2)

def norm():
    res = 0
    for times in range(2):
        for i in range(1000000):
            res += i + i ** 3 + i ** 2
    print('normal:',res)

if __name__=='__main__':
    st = time.time()
    norm()
    st1 = time.time()
    print('normal time:', st1 - st)
    multipro()
    st2 = time.time()
    print('multiprocessing time:', st2 -st1)
    multithread()
    st3= time.time()
    print('multithread time:', st3 - st2)