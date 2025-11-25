__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ._result import model_evaluation

__all__ = [model_evaluation]