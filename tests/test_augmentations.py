import pytest

from feral_segmentor.data.augmentations import Augmentation, FunctionAugmentation


class AppendTag(Augmentation):
    def __init__(self, tag: str, inner: Augmentation | None = None):
        super().__init__(inner)
        self.tag = tag

    def _apply(self, sample: list) -> list:
        return sample + [self.tag]

    def _own_params(self) -> dict:
        return {"tag": self.tag}


class Double(Augmentation):
    def _apply(self, sample: list) -> list:
        return sample + sample


def test_bare_augmentation_calls_apply_directly():
    aug = AppendTag("a")
    assert aug.augment([]) == ["a"]


def test_constructor_chaining_runs_inner_first():
    chain = AppendTag("outer", inner=AppendTag("inner"))
    assert chain.augment([]) == ["inner", "outer"]


def test_nested_chaining_three_deep():
    chain = AppendTag("c", inner=AppendTag("b", inner=AppendTag("a")))
    assert chain.augment([]) == ["a", "b", "c"]


def test_to_params_recurses_through_chain():
    chain = AppendTag("outer", inner=Double())
    assert chain.to_params() == {"0.AppendTag.tag": "outer"}


def test_to_params_indexes_same_class_repeated_in_chain():
    chain = AppendTag("outer", inner=AppendTag("inner"))
    assert chain.to_params() == {"0.AppendTag.tag": "outer", "1.AppendTag.tag": "inner"}


def test_variant_label_joins_chain_in_order():
    chain = AppendTag("outer", inner=Double())
    assert chain.variant_label() == "Double_AppendTag"


def test_function_augmentation_apply_raises():
    aug = FunctionAugmentation(fn=lambda x: x)
    with pytest.raises(NotImplementedError):
        aug.augment("sample")


def test_function_augmentation_params_expose_fn_name():
    aug = FunctionAugmentation(fn=lambda x: x)
    assert aug.to_params() == {"0.FunctionAugmentation.fn": "function"}


def test_function_augmentation_spreads_kwargs_into_params():
    aug = FunctionAugmentation(fn=lambda x: x, angle=30, p=0.5)
    assert aug.to_params() == {
        "0.FunctionAugmentation.fn": "function",
        "0.FunctionAugmentation.angle": 30,
        "0.FunctionAugmentation.p": 0.5,
    }
