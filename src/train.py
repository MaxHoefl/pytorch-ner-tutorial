""" 
Training the NER model
"""

import torch
import torch.optim as optim
from tqdm import trange


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    # set model to training mode
    model.train()

    for i in trange(num_steps):
        # fetch the next training batch
        train_batch, labels_bath = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_bath)

        # clear previous gradients, compute gradients wrt loss
        optimizer.zero_grad()
        loss.backward()

        # perform updates using calculated gradients
        optimizer.step()
   

if __name__ == '__main__':
    data_loader = DataLoader(data_dir)
