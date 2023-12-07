from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD
from decoder import decode
from utils import concat_inputs

import matplotlib.pyplot as plt
from dataloader import get_dataloader

def train(model, args):
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    val_loader = get_dataloader(args.val_json, args.batch_size, False)
    criterion = CTCLoss(zero_infinity=True)
    optimiser = SGD(model.parameters(), lr=args.lr)

    def train_one_epoch(epoch):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('checkpoints/{}'.format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_val_loss = 1e+6
    all_train_losses = []
    all_val_losses = []
    all_val_pers = []

    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.
        for idx, data in enumerate(val_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            outputs = log_softmax(model(inputs), dim=-1)
            val_loss = criterion(outputs, targets, in_lens, out_lens)
            running_val_loss += val_loss
        avg_val_loss = running_val_loss / len(val_loader)
        val_decode = decode(model, args, args.val_json)
        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, avg_val_loss, val_decode[4])
            )
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(avg_val_loss.item())
        all_val_pers.append(val_decode[4])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'checkpoints/{}/model_{}'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    print('train_loss')
    print(all_train_losses)
    print('valid_loss')
    print(all_val_losses)
    print('valid_per')
    print(all_val_pers)
    epochs = range(1, 21)

    # Plotting
    plt.figure(figsize=(15, 5))

    # Train Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, all_train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid(True)

    # Validation Loss Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, all_val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)

    # Validation PER Plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, all_val_pers, label='Validation PER', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('PER (%)')
    plt.title('Validation PER')
    plt.grid(True)

    # Save plot to a file
    plt.tight_layout()
    plt.savefig("num_layers"+str(args.num_layers)+'__clip_max_norm'+str(args.clip_max_norm)+'training_plots.png')

    # Save data to a file
    with open("num_layers"+str(args.num_layers)+'__clip_max_norm'+str(args.clip_max_norm)+'training_data.txt', 'w') as file:
        file.write('train_loss\n')
        file.write(str(all_train_losses) + '\n')
        file.write('valid_loss\n')
        file.write(str(all_val_losses) + '\n')
        file.write('valid_per\n')
        file.write(str(all_val_pers) + '\n')
    return model_path

