"""
test_feature_extractor.py
"""
import pytest
import torch
from unittest import mock
from pyslyde.encoders.feature_extractor import FeatureGenerator

# Fixtures
@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_path = tmp_path / "dummy.pt"
    dummy_state = {
        "state_dict": torch.nn.Sequential().state_dict(),
        "model": torch.nn.Sequential().state_dict(),
        "teacher": torch.nn.Sequential().state_dict()
    }
    torch.save(dummy_state, ckpt_path)
    return ckpt_path


@pytest.fixture
def fg(mock_checkpoint):
    return FeatureGenerator(model_name="moco", model_path=str(mock_checkpoint))


# Encoder Initialization Tests
@pytest.mark.parametrize("encoder_name", ["resnet18", "resnet50", "vgg16"])
def test_encoder_switches_correctly(mock_checkpoint, encoder_name):
    fg = FeatureGenerator(model_name="moco", model_path=str(mock_checkpoint), encoder_name=encoder_name)
    assert encoder_name in str(fg.encoder), f"Expected encoder '{encoder_name}' but got {type(fg.encoder)}"


def test_invalid_encoder_name_raises_keyerror(mock_checkpoint):
    with pytest.raises(KeyError):
        FeatureGenerator(model_name="moco", model_path=str(mock_checkpoint), encoder_name="invalid_encoder").encoder


# Model Loader Tests
@pytest.mark.parametrize(
    "model_name, loader_method",
    [
        ("moco", "_moco"),
        ("vgg16", "_vgg16"),
        ("ciga", "_ciga"),
        ("simclr", "_simclr"),
        ("hipt4k", "_hipt4k"),
        ("hipt256", "_hipt256"),
        ("transpath", "_transpath"),
        ("dinobrca", "_dinobrca"),
        ("uni", "_uni"),
        ("virchow2", "_virchow2"),
        ("gigapath", "_gigapath"),
        ("phikon", "_phikon"),
    ]
)
def test_model_loader_methods(model_name, loader_method, mock_checkpoint):
    if model_name == "vgg16":
        with mock.patch(
            "pyslyde.encoders.feature_extractor.models.vgg16",
            autospec=True,
        ) as mock_vgg, mock.patch(
            "pyslyde.encoders.feature_extractor.login", autospec=True
        ):
            mock_model = torch.nn.Sequential()
            mock_vgg.return_value = mock_model

            fg = FeatureGenerator(model_name="vgg16", model_path=str(mock_checkpoint))

            loader = getattr(fg, loader_method)
            model = loader()

            assert isinstance(model, torch.nn.Module)
            assert mock_vgg.called
            assert "weights" in mock_vgg.call_args.kwargs    
    elif model_name == "phikon":
        with mock.patch("pyslyde.encoders.feature_extractor.iBOTViT") as mock_ibotvit, \
             mock.patch("pyslyde.encoders.feature_extractor.login", autospec=True) as mock_login:
            mock_model = mock.Mock()
            mock_model.transform = mock.Mock()
            mock_model.to.return_value = mock_model
            mock_ibotvit.return_value = mock_model

            fg = FeatureGenerator(model_name=model_name, model_path=str(mock_checkpoint))
            loader = getattr(fg, loader_method)
            model = loader()
            assert hasattr(model, "to")
    else:
        with mock.patch("pyslyde.encoders.feature_extractor.torch.load") as mock_torch_load, \
             mock.patch("pyslyde.encoders.feature_extractor.timm.create_model", autospec=True) as mock_timm, \
             mock.patch("pyslyde.encoders.feature_extractor.login", autospec=True) as mock_login:

            mock_model = torch.nn.Sequential()
            mock_model.pretrained_cfg = {"input_size": (3, 224, 224)}
            mock_timm.return_value = mock_model

            if model_name in ["moco", "ciga"]:
                mock_torch_load.return_value = {"state_dict": torch.nn.Sequential().state_dict()}
            elif model_name in ["transpath"]:
                mock_torch_load.return_value = {"model": torch.nn.Sequential().state_dict()}
            else:
                mock_torch_load.return_value = torch.nn.Sequential().state_dict()

            fg = FeatureGenerator(model_name=model_name, model_path=str(mock_checkpoint))
            loader = getattr(fg, loader_method)
            model = loader()
            assert isinstance(model, torch.nn.Module)


# Initialization Behavior
def test_initialization_sets_model_and_name(fg):
    assert hasattr(fg, "model")
    assert fg.model_name == "moco"
    assert fg.model_path.endswith(".pt")


@mock.patch("torch.cuda.is_available", return_value=False)
def test_device_fallback_to_cpu(mock_cuda, mock_checkpoint): 
    fg = FeatureGenerator(model_name="moco", model_path=str(mock_checkpoint))
    assert fg.device == "cpu"


@mock.patch("pyslyde.encoders.feature_extractor.timm.create_model")
def test_timm_create_model_returns_module(mock_create_model, mock_checkpoint):
    mock_model = torch.nn.Sequential()
    mock_model.pretrained_cfg = {"input_size": (3, 224, 224)}
    mock_create_model.return_value = mock_model
    fg = FeatureGenerator(model_name="transpath", model_path=str(mock_checkpoint))
    model = fg._transpath()
    assert isinstance(model, torch.nn.Module)


# Forward Pass Test
@mock.patch("pyslyde.encoders.feature_extractor.Image.fromarray")
def test_forward_pass_returns_tensor(mock_fromarray, fg):
    mock_transform = mock.Mock()
    mock_transform.return_value = torch.randn(3, 224, 224)
    fg.transforms = mock_transform
    fg._model = mock.Mock(return_value=torch.randn(512))
    mock_fromarray.return_value = mock.Mock()
    dummy_image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
    output = fg.forward_pass(dummy_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (512,)


# Error Handling
def test_invalid_model_name_raises_attribute_error(mock_checkpoint):
    with pytest.raises(AttributeError):
        FeatureGenerator(model_name="unknown_model", model_path=str(mock_checkpoint)).model


@pytest.mark.parametrize("name", ["moco", "ciga", "simclr", "hipt256", "uni"])
@mock.patch("pyslyde.encoders.feature_extractor.torch.load", side_effect=FileNotFoundError)
def test_model_checkpoint_missing_raises_file_error(mock_load, name, tmp_path):
    with pytest.raises(FileNotFoundError):
        FeatureGenerator(model_name=name, model_path=str(tmp_path / "missing.pt"))