import multiprocessing as mp

def job(x):
    return x*x

def multiprocess():
    pool = mp.Pool() #自动分配到所有核进行运算，可以通过pool(processes=2),指定运用2个核
    res = pool.map(job,range(10000))
    print(res)


if __name__ == '__main__':
    multiprocess()
