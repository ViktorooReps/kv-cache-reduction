from typing import ContextManager, Optional, Callable


class WrappedManager(ContextManager):
    def __init__(self, to_wrap: Optional[ContextManager], on_exit: Optional[Callable]):
        self.wrapped = to_wrap
        self.on_exit = on_exit

    def __enter__(self):
        if self.wrapped is not None:
            self.wrapped.__enter__()

    def __exit__(self, __exc_type, __exc_value, __traceback):
        if self.wrapped is not None:
            self.wrapped.__exit__(__exc_type, __exc_value, __traceback)

        if __exc_type is not None:
            raise __exc_value

        if self.on_exit is not None:
            self.on_exit()

        return False