from gameframe.singleton import Singleton
from gameframe.vector import Vector2
from gameframe.sprite import Sprite
from gameframe.logger import Logger

class EventManager(Singleton): 
    def Init(self):
        self._evt = dict();

    def Register(event, callback): 
        if event not in self._evt: 
            self._evt[event] = [];
        self._evt[event].append(callback);

    def Unregister(event, callback): 
        if event not in self._evt: 
            return;

        evt_list = self._evt[event];
        for i in range(0, len(evt_list)): 
            if evt_list[i] == callback: 
                evt_list.pop(i);
                break;

    def Dispatch(event): 
        if event not in self._evt: 
            return;

        evt_list = self._evt[event];
        for callback in evt_list: 
            callback();