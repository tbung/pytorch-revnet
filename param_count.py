import models


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


print("revnet38: {}".format(get_param_size(models.revnet38())))
print("resnet32: {}".format(get_param_size(models.resnet32())))
