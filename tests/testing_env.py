from main import run
from ugfl.datasets import *

def test_main():
    run()
    return True

def test_dataloaders():
    data = load_real_dataset('Cora')
    dataloader, _ = create_federated_dataloader('Cora', ratio_train=0.4, ratio_test=0.2, n_clients=10)
    for batch in dataloader:
        print('working')
        break

    dataloader.current_client_id = 5
    for batch in dataloader:
        print('changed client')
        break

    dataloader.phase_mode = 'val'
    for batch in dataloader:
        print('changed phase of evaluation')
        break

    return True