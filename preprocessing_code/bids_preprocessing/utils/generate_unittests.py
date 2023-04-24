import os
import sys
import importlib.util
import typing
from typing import Callable

GENERAL_SKELETON = """import unittest

{classes}
"""

CLASS_SKELETON = """class {test_name}Test(unittest.TestCase):

{methods}
"""

METHOD_SKELETON = """    def test_{method_name}(self):
        pass

"""


def import_path(path):
    spec = importlib.util.spec_from_file_location("a_b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_classes(mod):
    md = mod.__dict__
    return [
        md[c]
        for c in md
        if (isinstance(md[c], type) and md[c].__module__ == mod.__name__)
    ]


def get_functions(mod):
    md = mod.__dict__
    return [
        md[c]
        for c in md
        if (
            isinstance(md[c], typing.Callable)
            and not isinstance(md[c], type)
            and md[c].__module__ == mod.__name__
        )
    ]


def get_methods(class_):
    return [
        func
        for func in dir(class_)
        if callable(getattr(class_, func)) and not func.startswith("_")
    ]


if __name__ == "__main__":

    folder = "/esat/spchtemp/scratch/baccou/bids_preprocessing/bids_preprocessing/preprocessing"  # os.path.abspath(sys.argv[1])
    test_folder = (
        "/esat/spchtemp/scratch/baccou/bids_preprocessing/tests/test_preprocessing"
    )
    for p in [
        "/users/spraak/spch/prog/spch/tensorflow_p3.6-2.3.0/lib/python3.6/site-packages",
        "/users/spraak/spch/prog/spch/TensorRT-6.0.1.5/lib/python3.6/site-packages",
        "/users/spraak/spchprog/SPRAAK/current/scripts",
        "/users/spraak/spch/prog/spch/morfessor-2.0.1",
    ]:
        if p in sys.path:
            sys.path.remove(p)
    print(sys.path)

    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if "__pycache__" in root:
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, folder)
            if rel_path != ".":
                path_parts = rel_path.split(os.sep)
                all_parts = ["test_" + x for x in path_parts + [filename]]
                test_path = os.path.join(test_folder, *all_parts)
            else:
                test_path = os.path.join(test_folder, "test_" + filename)
            # print(rel_path, all_parts, test_path)

            if path.startswith(test_folder):
                print(f"Not writing a test about a test ({test_path})")
                continue
            if os.path.exists(test_path):
                print(f"{test_path} already exists")
                continue

            try:
                mod = import_path(path)
            except Exception as e:
                print(e)
            functions = get_functions(mod)

            classes_str = ""

            for function in functions:
                function_name = function.__name__
                while "_" in function_name:
                    index = function_name.index("_")
                    temp = function_name[:index]
                    if index + 1 < len(function_name):
                        temp += (
                            function_name[index + 1].upper()
                            + function_name[index + 2 :]
                        )
                    function_name = temp
                method_str = METHOD_SKELETON.format(method_name=function_name)

                classes_str += CLASS_SKELETON.format(
                    test_name=function_name, methods=method_str
                )

            classes = get_classes(mod)
            for class_ in classes:
                methods_str = ""
                for method in get_methods(class_):
                    method_name = method
                    method_str = METHOD_SKELETON.format(method_name=method_name)
                    methods_str += method_str
                if len(methods_str) == 0:
                    methods_str = "    pass"
                classes_str += CLASS_SKELETON.format(
                    test_name=class_.__name__, methods=methods_str
                )

            full_file = GENERAL_SKELETON.format(classes=classes_str)

            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            print(f"Writing {test_path}")
            with open(test_path, "w") as fp:
                fp.write(full_file)
