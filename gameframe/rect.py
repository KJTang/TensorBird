
class Rect(): 
    _x = 0;
    _y = 0;
    _width = 0;
    _height = 0;

    def __init__(self, x = 0, y = 0, width = 0, height = 0): 
        self._x = x;
        self._y = y;
        self._width = width;
        self._height = height;

    def __str__(self): 
        return "Rect: (" + str(self._x) + ", " + str(self._y) + ", " + str(self._width) + ", " + str(self._height) + ")";

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

    @property
    def width(self):
        return self._width;
        
    @width.setter
    def width(self, val):
        self._width = val;

    @property
    def height(self):
        return self._height;
        
    @height.setter
    def height(self, val):
        self._height = val;

    @staticmethod
    def isOverlapPoint(rect, point): 
        if (point.x >= rect.x) and (point.x <= rect.x + rect.width) and (point.y >= rect.y) and (point.y <= rect.y + rect.height): 
            return True;
        return False;

    @staticmethod
    def isOverlapRect(recta, rectb): 
        if (recta.x <= rectb.x + rectb.width) and (recta.y <= rectb.y + rectb.height) and (rectb.x <= recta.x + recta.width) and (rectb.y <= recta.y + recta.height): 
            return True;
        return False;
