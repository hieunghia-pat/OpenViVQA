from typing import Any, Dict, List, OrderedDict, Union
import torch
import numpy as np

from .logging_utils import setup_logger

logger = setup_logger()

class Instance(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"{key} not found")

    def get_fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())

class InstanceList(OrderedDict):
    def __init__(self, instance_list: List["Instance"] = []):
        super().__init__(self)

        if len(instance_list) == 0:
            return

        assert all(isinstance(i, Instance) for i in instance_list)

        for key in instance_list[0].get_fields():
            values = [instance.get(key) for instance in instance_list]
            v0 = values[0]
            if isinstance(v0, np.ndarray):
                values = [torch.tensor(value) for value in values]
                values = self.pad_values(values)
                values = torch.cat(values, dim=0)
            if isinstance(v0, torch.Tensor):
                values = self.pad_values(values)
                values = torch.cat(values, dim=0)
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                values
            self.set(key, values)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self:
            raise AttributeError(f"{name} not found")
        return self[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of Instance,
        and must agree with other existing fields in this object.
        """
        self[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return list(self.keys())

    @property
    def batch_size(self) -> int:
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                return self[k].shape[0]
            if isinstance(self[k], list):
                return len(self[k])

        return 0

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "InstanceList":
        """
        Returns:
            InstanceList: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = InstanceList()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    # Tensor-like methods
    def unsqueeze(self, *args: Any, **kwargs) -> "InstanceList":
        """
        Returns:
            InstanceList: all fields are called with a `unsqueeze(dim)`, if the field has this method.
        """
        ret = InstanceList()
        for k, v in self.items():
            if hasattr(v, "unsqueeze"):
                v = v.unsqueeze(*args, **kwargs)
            ret.set(k, v)
        
        return ret

    # Tensor-like methods
    def squeeze(self, *args: Any, **kwargs) -> "InstanceList":
        """
        Returns:
            InstanceList: all fields are called with a `squeeze(dim)`, if the field has this method.
        """
        ret = InstanceList()
        for k, v in self.items():
            if hasattr(v, "squeeze"):
                v = v.squeeze(*args, **kwargs)
            ret.set(k, v)
        
        return ret

    # special method for concatenating tensor objects
    def pad_values(self, values: List[torch.tensor], padding_value=0) -> List[torch.tensor]:
        padded_values = []
        max_len = max([value.shape[0] for value in values])
        for value in values:
            additional_len = max_len - value.shape[0]
            
            if additional_len == 0:
                padded_values.append(value.unsqueeze(0))
                continue

            padding_tensor = torch.zeros((additional_len, value.shape[-1])).fill_(padding_value)
            value = torch.cat([value, padding_tensor], dim=0)

            padded_values.append(value.unsqueeze(0))
        
        return padded_values

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self.items())))
        return s

    __repr__ = __str__