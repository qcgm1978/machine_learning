class DP(object):
    two_count = 0
    m={'0':0,'1':1}

    def fib(self, n):
        self.two_count = 0
        val = self.get_fib(n)
        return val
    # requires only O(n) time instead of exponential time (but requires O(n) space)
    def get_fib(self,n):
        s_n=str(n)
        if s_n not in self.m:
            if n==2:
                self.two_count += 1
            p = self.get_fib(n - 1)
            q = self.get_fib(n - 2)
            self.m[s_n]=p+q
        return self.m[s_n]
