import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    target_model_name = model_name.replace('_', '') + 'model'
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print("No subclass of BaseModel matching '%s' found in %s." % (target_model_name, model_filename))
        exit(0)
    return model


def get_option_setter(model_name):
    return find_model_using_name(model_name).modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % instance.name())
    return instance
