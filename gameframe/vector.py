
class Vector2(): 
    _x = 0;
    _y = 0;

    def __init__(self, x = 0, y = 0): 
        self._x = x;
        self._y = y;

    def __str__(self): 
        return "Vector2: (" + str(self._x) + ", " + str(self._y) + ")";

    @property
    def x(self):
        return self._x;
        
    @x.setter
    def x(self, val):
        self._x = val;

    @property
    def y(self):
        return self._y;
        
    @y.setter
    def y(self, val):
        self._y = val;

    @staticmethod
    def Add(a, b): 
        return Vector2(a.x + b.x, a.y + b.y);

    @staticmethod
    def Sub(a, b): 
        return Vector2(a.x - b.x, a.y - b.y);