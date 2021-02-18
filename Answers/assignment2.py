class calculator:
    def __init__(self):
        self.a = 1
        self.b = 1
        self.ans = -1

    def inp(self):
        expr = input("Enter the expression to be calculated : ")
        expr = expr.strip()
        if expr.find('+') != -1:
            [self.a,self.b] = expr.split('+')
            self.a = int(self.a)
            self.b = int(self.b)
            self.ans = self.a + self.b
        elif expr.find('-') != -1:
            [self.a,self.b] = expr.split('-')
            self.a = int(self.a)
            self.b = int(self.b)
            self.ans = self.a - self.b
        elif expr.find('*') != -1:
            [self.a,self.b] = expr.split('*')
            self.a = int(self.a)
            self.b = int(self.b)
            self.ans = self.a * self.b
        elif expr.find('//') != -1:
            [self.a,self.b] = expr.split('//')
            self.a = int(self.a)
            self.b = int(self.b)
            self.ans = self.a // self.b
        elif expr.find('/') != -1:
            [self.a,self.b] = expr.split('/')
            self.a = float(self.a)
            self.b = float(self.b)
            self.ans = float(self.a / self.b)
        elif expr.find('%') != -1:
            [self.a,self.b] = expr.split('%')
            self.a = int(self.a)
            self.b = int(self.b)
            self.ans = self.a % self.b

    def out(self):
        print("Final result :",self.ans)

calc = calculator()
calc.inp()
calc.out()
