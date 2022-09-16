import itertools
from typing import Any, Dict, List, Union
import torch

from .logging_utils import setup_logger

logger = setup_logger()

# special method for concatenating tensor objects
def pad_values(values: List[torch.tensor]) -> List[torch.tensor]:
        padded_values = []
        max_len = max([value.shape[1] for value in values])
        for value in values:
            additional_len = max_len - value.shape[1]
            
            if additional_len == 0:
                padded_values.append(value)
                continue

            if len(value.shape) > 2:
                padding_value = torch.zeros((value.shape[0], additional_len, value.shape[-1]))
            else:
                padding_value = torch.zeros((value.shape[0], additional_len))
            value = torch.cat([value, padding_value], dim=1)

            padded_values.append(value)
        
        return padded_values

class Instances:
    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: fields to add to this `Instances`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            logger.warning("Cannot find field '{}' in the given Instances!".format(name))
            return None
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        if not hasattr(self, "_batch_size"):
            if isinstance(value, torch.tensor):
                self._batch_size = value.shape[0]
            elif isinstance(value, list):
                self._batch_size = len(value)
        else:
            assert self._batch_size == (value.shape[0] if isinstance(value, torch.tensor) else len(value)), f"{name} has different batch size!"
        self._fields[name] = value

    @property
    def batch_size(self) -> int:
        if hasattr(self, "_batch_size"):
            return self._batch_size
        else:
            return 0

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances()
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Instances()
        for key in instance_lists[0]._fields.keys():
            values = [instance.get(key) for instance in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = pad_values(values)
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                values
            ret.set(key, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
