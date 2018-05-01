import pygame
from gameframe.vector import Vector2
from gameframe.logger import Logger

logger = Logger();

class Sprite(): 
    _img = None;

    def __init__(self, path = ""):
        self._parent = None;
        self._children = [];

        self._is_pos_dirty = True;
        self._local_position = Vector2();
        self._position = Vector2();

        self._size = Vector2();
        if (not path is None) and (path != ""): 
            self._img = pygame.image.load(path)
            self._size = Vector2(self._img.get_width(), self._img.get_height());

    def Update(self): 
        pass;

    def Render(self, screen): 
        if not self._img is None: 
            screen.blit(self._img, (self.Position.x, self.Position.y))

    def AddChild(self, child): 
        self._children.append(child);
        child.Parent = self;

    def RemoveChild(self, child, destroy = True): 
        for i in range(0, len(self._children)): 
            if self._children[i] == child: 
                self._children.pop(i);
                if destroy: 
                    del child;
                break;

    def GetChildren(self): 
        return self._children;

    @property
    def Parent(self):
        return self._parent;
        
    @Parent.setter
    def Parent(self, val):
        self._parent = val;

    ''' world position: read only '''
    @property
    def Position(self):
        if self._is_pos_dirty: 
            self.RecalculateWorldPosition();
        return self._position;

    def RecalculateWorldPosition(self):
        if not self.Parent is None: 
            self._position = Vector2.Add(self.LocalPosition, self.Parent.Position);
        else:
            self._position = self.LocalPosition;

        self._is_pos_dirty = False;

    ''' local position '''
    @property
    def LocalPosition(self):
        return self._local_position;

    @LocalPosition.setter
    def LocalPosition(self, val): 
        self._local_position = val;
        self._is_pos_dirty = True;


    @property
    def Size(self):
        return self._size;

    @property
    def Width(self):
        return self._size.x;

    @property
    def Height(self):
        return self._size.y;

    @property
    def Rect(self): 
        if self._is_pos_dirty: 
            self._rect = pygame.Rect(self.Position.x, self.Position.y, self.Width, self.Height);
        return self._rect;

