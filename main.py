import yaml
from data import load_data
from model import TransBathy


def run():
    # Loading configurations in yaml file
    with open('config/config-cpu.yaml', 'r') as file:
        config = yaml.safe_load(file)
    phase = config['Phase']

    # build dataloader
    dataloader = load_data(config)

    # build model
    model = TransBathy(config=config, dataloader=dataloader)

    if phase == 'train':
        # start training
        model.train()

    elif phase == 'test':
        # star testing
        results = model.test()

        print(">> Testing Done, {} is {}".format(config['Metric'], results[config['Metric']]))


if __name__ == '__main__':
    run()
