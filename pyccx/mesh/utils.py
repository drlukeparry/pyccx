class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Ent:
    """
    Entity types for Elementary MSH Geometry
    """
    Point = 0
    Curve = 1
    Surface = 2
    Volume = 3
    All = -1
