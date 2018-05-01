# from gameframe.singleton import Singleton
# from gameframe.vector import Vector2
# from gameframe.sprite import Sprite
# from gameframe.logger import Logger

# class PhysicsManager(Singleton): 
#     def Init(self):
#         self._groups = set();
#         self._rules = set();

#     def Register(group, sprite): 
#         if group not in self._groups: 
#             self._groups[group] = [];
#         self._groups[group].append(sprite);

#     def Unregister(group, sprite): 
#         if group not in self._groups: 
#             return;

#         sprite_list = self._groups[group];
#         for i in range(0, len(sprite_list)): 
#             if sprite_list[i] == sprite: 
#                 sprite_list.pop(i);
#                 break;

#     def AddRule(a, b): 
#         if a not in self._rules: 
#             self._rules[a] = [];
#         self._rules[a].append(b);

#     def RemoveRule(a, b): 
#         if a not in self._rules: 
#             return;

#         rule_list = self._rules[a];
#         for i in range(0, len(rule_list)): 
#             if rule_list[i] == b: 
#                 rule_list.pop(i):
#                 break;

#     def IsRuleExist(a, b): 
#         if a in self._rules: 
#             rule_list = self._rules[a];
#             for i in range(0, len(rule_list)): 
#                 if rule_list[i] == b: 
#                     return True;
#         if b in self._rules: 
#             rule_list = self._rules[b];
#             for i in range(0, len(rule_list)): 
#                 if rule_list[i] == a: 
#                     return True;
#         return False;

#     def Update(self): 
#         