"""Tiny stand-in for Streamlit's @st.cache_data so the engines work without opening a screen."""
import functools, time


def ttl_cache(ttl=3600, **_ignored):
    def deco(fn):
        store = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            try:
                hit = store.get(key)
            except TypeError:                      # unhashable args: just run it
                return fn(*args, **kwargs)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
            val = fn(*args, **kwargs)
            store[key] = (time.time(), val)
            return val
        wrapper.clear = store.clear
        return wrapper
    return deco


def clear_all(*modules):
    for m in modules:
        for v in vars(m).values():
            if callable(getattr(v, "clear", None)) and hasattr(v, "__wrapped__"):
                v.clear()
