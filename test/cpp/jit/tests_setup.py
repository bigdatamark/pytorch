# [MODULE DOC] Helper that creates and removes fixture files needed by the C++
# JIT tests. Defines setup objects (each creates one saved model/value file in
# the current directory and removes it on shutdown) and a list of them. Run as
# a script with one argument, setup or shutdown, to apply that action to all
# fixtures; any other argument is silently ignored and a missing argument
# raises IndexError. No import-time side effects beyond definitions.
import os
import sys

import torch


class Setup:
    # [CLASS DOC] HELPER: Abstract interface for a fixture with setup and shutdown
    # actions.
    def setup(self):
        # [FUNCTION DOC] HELPER: Abstract; subclasses create the fixture. Raises
        # NotImplementedError.
        raise NotImplementedError

    def shutdown(self):
        # [FUNCTION DOC] HELPER: Abstract; subclasses remove the fixture. Raises
        # NotImplementedError.
        raise NotImplementedError


class FileSetup:
    # [CLASS DOC] HELPER: Fixture base class that owns one file at the class
    # attribute path (None by default); shutdown deletes that file. Subclasses set
    # path and implement setup to write it.
    path = None

    def shutdown(self):
        # [FUNCTION DOC] HELPER: Deletes the file at path if it exists; does nothing
        # otherwise.
        if os.path.exists(self.path):
            os.remove(self.path)


class EvalModeForLoadedModule(FileSetup):
    # [CLASS DOC] HELPER: Fixture that saves a scripted model with dropout, left
    # in training mode, to dropout_model.pt, so the C++ test can check the mode is
    # restored as eval on load.
    path = "dropout_model.pt"

    def setup(self):
        # [FUNCTION DOC] HELPER: Defines a script module applying dropout (p 0.1),
        # puts it in train mode and saves it to path.
        class Model(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Local scripted module with a single dropout layer
            # applied in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dropout layer with probability 0.1.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.1)

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies dropout to x and returns the result.
                x = self.dropout(x)
                return x

        model = Model()
        model = model.train()
        model.save(self.path)


class SerializationInterop(FileSetup):
    # [CLASS DOC] HELPER: Fixture that saves a pair of tensors with the new zip
    # based serialization to ivalue.pt, for C++ serialization interop tests.
    path = "ivalue.pt"

    def setup(self):
        # [FUNCTION DOC] HELPER: Saves a tuple of a 2x2 tensor of ones and a 3x5
        # tensor of twos to path using the zip file format.
        ones = torch.ones(2, 2)
        twos = torch.ones(3, 5) * 2

        value = (ones, twos)

        torch.save(value, self.path, _use_new_zipfile_serialization=True)


# See testTorchSaveError in test/cpp/jit/tests.h for usage
class TorchSaveError(FileSetup):
    # [CLASS DOC] HELPER: Fixture that saves a pair of tensors with the legacy
    # (non-zip) format to eager_value.pt, used by the C++ test that expects a load
    # error.
    path = "eager_value.pt"

    def setup(self):
        # [FUNCTION DOC] HELPER: Saves a tuple of a 2x2 tensor of ones and a 3x5
        # tensor of twos to path using the legacy non-zip file format.
        ones = torch.ones(2, 2)
        twos = torch.ones(3, 5) * 2

        value = (ones, twos)

        torch.save(value, self.path, _use_new_zipfile_serialization=False)


class TorchSaveJitStream_CUDA(FileSetup):
    # [CLASS DOC] HELPER: Fixture that saves a scripted model using cuda streams
    # to saved_stream_model.pt; the file is produced only when cuda is available.
    path = "saved_stream_model.pt"

    def setup(self):
        # [FUNCTION DOC] HELPER: Returns without doing anything if cuda is
        # unavailable. Otherwise scripts a model that creates a cuda stream and
        # tensors, concatenates them on that stream and returns the stream check with
        # the tensors, and saves it to path.
        if not torch.cuda.is_available():
            return

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward runs a cat on a fresh cuda
            # stream.
            def forward(self):
                # [FUNCTION DOC] HELPER: Creates a stream and two random cuda tensors,
                # concatenates them inside the stream context, synchronizes, and returns
                # whether the current stream matched, a, b and the cat.
                s = torch.cuda.Stream()
                a = torch.rand(3, 4, device="cuda")
                b = torch.rand(3, 4, device="cuda")

                with torch.cuda.stream(s):
                    is_stream_s = (
                        torch.cuda.current_stream(s.device_index()).id() == s.id()
                    )
                    c = torch.cat((a, b), 0).to("cuda")
                s.synchronize()
                return is_stream_s, a, b, c

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        torch.jit.save(script_model, self.path)


tests = [
    EvalModeForLoadedModule(),
    SerializationInterop(),
    TorchSaveError(),
    TorchSaveJitStream_CUDA(),
]


def setup():
    # [FUNCTION DOC] HELPER: Runs setup on every fixture in the module list, in
    # order.
    for test in tests:
        test.setup()


def shutdown():
    # [FUNCTION DOC] HELPER: Runs shutdown on every fixture in the module list, in
    # order.
    for test in tests:
        test.shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "shutdown":
        shutdown()
