import math

class Value():
    def __init__(self,value,_children=(),_op="", label=""):
        self.data=value
        self.label=label
        self._prev=set(_children)
        self._op=_op
        self._grad=0.0
        self._backward=lambda:None
        
    def __repr__(self):
        return f"Value(data={self.data},{self.label})"
    
    def __add__(self,other):
        other = Value(other) if not isinstance(other,Value) else other
        out = Value(self.data+other.data,(self,other),"+")
        def _backward():
            self._grad += 1.0 * out._grad
            other._grad += 1.0 * out._grad
        out._backward=_backward
        return out
        
    def __mul__(self,other):
        other = Value(other) if not isinstance(other,Value) else other
        out = Value(self.data*other.data,(self,other),"*")
        
        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad
        out._backward=_backward
        return out

    def __pow__(self,val):
        assert isinstance(val, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**val,(self,),f"**{val}")
        
        def _backward():
            self._grad += (val*(self.data**(val-1))) * out._grad
        out._backward=_backward
        return out
        
    def relu(self):
        out = Value(max(0,self.data),(self,),"relu")
        
        def _backward():
            self._grad += (0 if self.data<=0 else 1) * out._grad
        out._backward=_backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self._grad += (1-(out.data**2)) * out._grad
        out._backward=_backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
          self.grad += out.data * out.grad
        out._backward = _backward
        return out


    def backward(self):
        nodes=[]
        visited=set([])
        def recc(node):
            if node in visited:
                return
            visited.add(node)
            for i in node._prev:
                recc(i)
            nodes.append(node)
        recc(self)
        self._grad=1.0
        for i in nodes[::-1]:
            i._backward()
        return
    def __sub__(self,other):
        return self.__add__(-(other))

    def __rsub__(self,other):
        return self.__add__(-(other))
        
    def __radd__(self,other):
        return self.__add__(other)

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return self * (-1)

    def __truediv__(self, other): 
        return self * other**-1