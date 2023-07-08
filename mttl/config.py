import json
import os
import ast
import argparse
from string import Template
from abc import abstractmethod, ABC


class Config(ABC):
    @abstractmethod
    def _set_defaults(self):
        pass

    def __init__(self, filenames=None, kwargs=None, raise_error=True):
        # Stores personalization of the config file in a dict (json serializable)
        self._updated_kwargs = {}
        self.filenames = filenames
        self._set_defaults()

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False, raise_error=raise_error)

        if kwargs:
            self.update_kwargs(kwargs, raise_error=raise_error)

        self.save_config(self.output_dir)

    def was_overridden(self, key):
        return key in self._updated_kwargs

    def was_default(self, key):
        return key not in self._updated_kwargs

    def update_kwargs(self, kwargs, eval=True, raise_error=True):
        for (k, v) in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    v = v
            else:
                v = v
            if not hasattr(self, k) and raise_error:
                raise ValueError(f"{k} is not in the config")

            if eval:
                print("Overwriting {} to {}".format(k, v))

            if k == 'finegrained':
                k = 'poly_granularity'
                v = 'finegrained' if v else 'coarsegrained'
            elif k in ['train_dir', 'output_dir']:
                # this raises an error if the env. var does not exist
                v = Template(v).substitute(os.environ)

            setattr(self, k, v)
            self._updated_kwargs[k] = v

    def __getitem__(self, item):
        return getattr(self, item, None)

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        import copy

        to_save = copy.deepcopy(self.__dict__)
        to_save.pop("_updated_kwargs")

        return json.dumps(to_save, indent=4, sort_keys=False)

    def save_config(self, output_dir):
        """
        Saves the config
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")
        
    @classmethod    
    def parse(cls, extra_kwargs=None, raise_error=True):
        import itertools

        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config_files", required=False)
        parser.add_argument("-k", "--kwargs", nargs="*", action='append')
        args = parser.parse_args()

        kwargs = {}
        if args.kwargs:
            kwargs_opts = list(itertools.chain(*args.kwargs))
            for value in kwargs_opts:
                key, _, value = value.partition('=')
                kwargs[key] = value
        args.kwargs = kwargs
        if extra_kwargs:
            args.kwargs.update(extra_kwargs)

        config = cls(args.config_files, args.kwargs, raise_error=raise_error)

        print(config.to_json())
        return config


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
