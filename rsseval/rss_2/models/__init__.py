
import os
import importlib

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]
names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, encoder, decoder, n_images, c_split):
    if args.model in ['cext', 'mnistdpl', 'mnistnn', 'mnistsl', 'mnistltn']:
        return names[args.model](encoder, n_images=n_images, c_split=c_split, args=args) # only discriminative
    else:
        return names[args.model](encoder, decoder, n_images=n_images, c_split=c_split, args=args)