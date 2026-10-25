# [MODULE DOC] Defines the common base class (and metaclass) for custom class
# objects, whose instance check delegates to the opaque-object registration
# system and also recognizes fake script object wrappers. Also exposes backward
# compatible aliases OpaqueBaseMeta and OpaqueBase. Import-time state: a module
# global cache for the fake script object type, filled lazily.
# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None


class CustomClassBaseMeta(type):
    # [CLASS DOC] Metaclass customizing instance checks for custom class types.
    # Stateless; the only method is the instance check override.
    def __instancecheck__(cls, instance):
        # [FUNCTION DOC] Instance check hook. For the base class itself, returns
        # whether the object is any registered custom class object (delegated to
        # the opaque object registry, imported lazily). For subclasses: true on
        # a normal match, else true if the object is a fake script object whose
        # wrapped real object matches. Fake type is imported lazily and cached in
        # a module global (global mutation). Returns a bool; avoids attribute
        # access on arbitrary objects until the fake type check passes.
        # When checking against CustomClassBase itself (not a concrete subclass),
        # delegate to the registration system which correctly covers all
        # custom classes (constant types, metaclass-only symbolic types, and
        # FakeScriptObject wrappers).
        if cls is CustomClassBase:
            from torch._library.opaque_object import is_custom_class_obj

            return is_custom_class_obj(instance)

        if super().__instancecheck__(instance):
            return True

        # Check FakeScriptObject before hasattr to avoid triggering custom
        # __getattr__ on arbitrary user objects (e.g. dict-like objects that
        # raise KeyError on unknown attributes).
        # e.g. test/dynamo/test_dynamic_shapes.py -k test_user_getattr1_dynamic_shapes
        global _FakeScriptObject_cls
        if _FakeScriptObject_cls is None:
            from torch._library.fake_class_registry import FakeScriptObject

            _FakeScriptObject_cls = FakeScriptObject
        if isinstance(instance, _FakeScriptObject_cls) and hasattr(
            instance, "real_obj"
        ):
            return super().__instancecheck__(instance.real_obj)

        return False


class CustomClassBase(metaclass=CustomClassBaseMeta):
    # [CLASS DOC] Marker base class for custom (opaque) class objects. Has no
    # state or methods; its metaclass makes isinstance against it true for any
    # registered custom class object, not only real subclasses.
    pass


# Backward compatibility aliases
OpaqueBaseMeta = CustomClassBaseMeta
OpaqueBase = CustomClassBase
