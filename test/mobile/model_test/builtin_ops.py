# [MODULE DOC] Mobile model test source defining two modules that exercise TorchScript builtin
# functions: scalar and tensor conversions, arithmetic and bitwise operators, comparisons, min and
# max, string methods, length queries, list and dictionary operations. They are scripted and
# exported by the test model generator to measure operator coverage. No import-time side effects.
import torch


# https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions


class TSBuiltinOpsModule(torch.nn.Module):
    # [CLASS DOC] Stateless module whose forward evaluates a wide set of builtin type, math,
    # comparison and string operations on small constant tensors, numbers, a string list and
    # two dictionaries.
    def forward(self):
        # [FUNCTION DOC] Creates sample tensors, a float, a list of strings and two dictionaries,
        # then passes the results of many builtin expressions (conversions, bit operations,
        # arithmetic, comparisons, min and max, string methods, length queries) as arguments to a
        # single length call and returns its result.
        # KNOWN ISSUE: the length builtin takes a single argument, so this forward is only
        #     meaningful when scripted (to collect operator names) and would raise a type error if
        #     run in eager mode.
        x = torch.tensor(1)
        y = torch.tensor(0.5)
        b = float(1)
        l = ["1", "2", "test", "a{}b"]
        d = {"key": 1}
        d2 = {0: 100}
        return len(
            # type
            bool(x),
            bool(x.item()),
            int(y),
            int(y.item()),
            float(x),
            float(x.item()),
            # math
            x & x,
            bool(x) & bool(x),
            int(x) & int(x),
            x | x,
            bool(x) | bool(x),
            int(x) | int(x),
            x << x,
            int(x) << int(x),
            x >> x,
            int(x) >> int(x),
            x ^ x,
            bool(x) ^ bool(x),
            int(x) ^ int(x),
            b * float(x),
            b * int(x),
            b + float(x),
            b - float(x),
            x.item() + y.item(),
            x.item() - y.item(),
            x.item() * y.item(),
            x.item() / y.item(),
            float(x) < float(y),
            float(x) <= float(y),
            float(x) > float(y),
            float(x) > int(y),
            float(x) >= float(y),
            float(x) >= int(y),
            float(x) == float(y),
            float(x) == int(y),
            float(x) != float(y),
            int(x) != float(y),
            float(x) / float(y),
            int(x) / int(y),
            max(x),
            max(x.item(), y.item()),
            max(int(x), int(y)),
            max(float(x), float(y)),
            min(x),
            min(x.item(), y.item()),
            min(int(x), int(y)),
            min(float(x), float(y)),
            int(l[0]),
            float(l[0]),
            # string
            str(torch.tensor(1)),
            l[2].find("t"),
            l[2].replace("t", "x"),
            l[2].lower(),
            l[2].startswith("t"),
            l[2].split("t"),
            l[2].strip(),
            l[2].rstrip(),
            l[2].lstrip(),
            l[2][slice(2)],
            l[3].format("x"),
            ord(l[2][0]),
            len(torch.randn(3)),
            len(l),
            len(l[2]),
            len(d),
            len(d2),
        )


class TSCollectionOpsModule(torch.nn.Module):
    # [CLASS DOC] Stateless module whose forward exercises list and dictionary builtins.
    def forward(self):
        # [FUNCTION DOC] Builds a string, a string list (reversed twice, element assigned,
        # extended), a string-keyed dictionary (cleared, updated, key checked and reassigned) and
        # an int-keyed dictionary (key checked, cleared, reassigned), then passes string indexing
        # by tensor, dictionary lookups, keys, items, values and a list pop as arguments to a
        # single length call and returns its result.
        # KNOWN ISSUE: the length builtin takes a single argument, so this forward is only
        #     meaningful when scripted (to collect operator names) and would raise a type error if
        #     run in eager mode.
        s = "abcde"
        # list
        l = ["1", "2", "test"]
        l.reverse()
        l.reverse()
        l[1] = "3"
        l.extend(["4"])
        # str dict
        d = {"key": 1}
        d.clear()
        d.update({"key": 0})
        if "key" in d:
            d["key"] = 2
        #  int dict
        d2 = {0: 100}
        if 0 in d2:
            d2.clear()
            d2[0] = 100

        return len(
            s[torch.tensor(1)],
            d["key"],
            d2[0],
            d.keys(),
            d.items(),
            d.values(),
            d2.values(),
            l.pop(),
        )
