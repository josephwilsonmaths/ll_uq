#--- Get hyperparameters from config file
import configparser

def get_config(fname, model, dataset):
    config = configparser.ConfigParser()
    config.read(fname)
    field = f'{model}_{dataset}'
    config_dict = {}
    config_dict['n_experiment'] = config.getint(field,'n_experiment')
    config_dict['epochs'] = config.getint(field,'epochs')
    config_dict['lr'] = config.getfloat(field,'lr')
    config_dict['wd'] = config.getfloat(field,'wd')
    config_dict['bs'] = config.getint(field,'bs')
    config_dict['optim'] = config.get(field,'optim')
    config_dict['sched'] = config.get(field,'sched')
    config_dict['S'] = config.getint(field,'S')
    config_dict['dnn_S'] = config.getint(field,'dnn_S')
    config_dict['dnn_epoch'] = config.getint(field,'dnn_epoch')
    config_dict['dnn_lr'] = config.getfloat(field,'dnn_lr')
    config_dict['dnn_bs'] = config.getint(field,'dnn_bs')
    config_dict['dnn_gamma'] = config.getfloat(field,'dnn_gamma')
    config_dict['ll_S'] = config.getint(field,'ll_S')
    config_dict['ll_epoch'] = config.getint(field,'ll_epoch')
    config_dict['ll_lr'] = config.getfloat(field,'ll_lr')
    config_dict['ll_bs'] = config.getint(field,'ll_bs')
    config_dict['ll_gamma'] = config.getfloat(field,'ll_gamma')
    return config_dict

