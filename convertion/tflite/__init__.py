from abc import ABC, abstractmethod
import tensorflow as tf


class SaveableModel(ABC):
    @abstractmethod
    def get_model_input(self) -> list[tf.keras.Input]:
        pass
